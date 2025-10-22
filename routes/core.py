import time
import uuid
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app import (
    AutoMemActionRequest,
    BOOKS,
    CONVERSATIONS,
    SUMMARIES,
    INSTRUCTIONS,
    MEMORIES,
    MEMORY_USE_RECENCY,
    PREFERENCES,
    SNAPSHOTS,
    autom,
    build_chapter_context,
    build_context,
    build_snapshot,
    call_openai_chat,
    detect_conv_register,
    detect_mode,
    logger,
    mark_facets_used,
    maybe_refresh_summary,
    render_system_prompt_author,
    render_system_prompt_conversation,
    strip_mode_marker,
    _persist_chapter_from_output,
    embed,
)
from models import (
    Book,
    ChatRequest,
    ChatResponse,
    InstructionOverride,
    Message,
    MemoryItem,
    SeedMemoryRequest,
    SeedPrefRequest,
    SetSnapshotRequest,
)
from stores import CHAPTERS


router = APIRouter(tags=["core"])


@router.get("/api/conversations/{session_id}")
def get_conversation(session_id: str, limit: int = 50, include_instructions: bool = False):
    """Return the latest conversation messages for a session."""

    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")

    convo = CONVERSATIONS.get(session_id, [])

    if not include_instructions:
        convo = [m for m in convo if getattr(m, "mode", None) != "instruction"]

    if limit and len(convo) > limit:
        convo = convo[-limit:]

    serialized = []
    for msg in convo:
        if hasattr(msg, "model_dump"):
            data = msg.model_dump()
        elif hasattr(msg, "dict"):
            data = msg.dict()
        else:
            data = {
                "role": getattr(msg, "role", ""),
                "content": getattr(msg, "content", ""),
                "ts": getattr(msg, "ts", None),
                "mode": getattr(msg, "mode", None),
            }
        serialized.append(
            {
                "role": data.get("role"),
                "content": data.get("content"),
                "ts": data.get("ts"),
                "mode": data.get("mode"),
            }
        )

    return {
        "ok": True,
        "items": serialized,
        "summary": SUMMARIES.get(session_id),
    }


@router.get("/api/memories/auto/pending")
def get_auto_pending(user_id: str):
    items = autom.list_pending(user_id)
    logger.info("auto_mem_pending_list", user_id=user_id, count=len(items))
    return {"ok": True, "items": [c.dict() for c in items]}


@router.post("/api/memories/auto/accept")
def post_auto_accept(req: AutoMemActionRequest):
    result = autom.accept_pending(req.user_id, req.ids)
    logger.info(
        "auto_mem_pending_accept",
        user_id=req.user_id,
        requested_ids=req.ids,
        accepted=result.get("accepted", 0),
        accepted_ids=result.get("accepted_ids", []),
    )
    return result


@router.post("/api/memories/auto/reject")
def post_auto_reject(req: AutoMemActionRequest):
    result = autom.reject_pending(req.user_id, req.ids)
    logger.info(
        "auto_mem_pending_reject",
        user_id=req.user_id,
        requested_ids=req.ids,
        deleted=result.get("deleted", 0),
    )
    return result


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    book = BOOKS.get(req.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    user_mode = detect_mode(req.message, default_mode="conversation")
    raw_user_text = strip_mode_marker(req.message)

    if user_mode == "instruction":
        # 1) store raw instruction (verbatim) in INSTRUCTIONS
        instr_list = INSTRUCTIONS.setdefault(req.user_id, [])
        instr_list.append(
            InstructionOverride(
                rule_key="raw",
                rule_value=req.message[2:].strip(),
                active=True,
            )
        )

        # 2) garder une trace dans CONVERSATIONS pour l’audit (tu filtres déjà ces messages du contexte)
        CONVERSATIONS.setdefault(req.session_id, []).append(
            Message(role="user", content=req.message, mode=user_mode)
        )

        # 3) important: ne pas appeler embed(), ne pas toucher MEMORIES
        return ChatResponse(output="", mode="instruction", used_facets=[], used_memory_ids=[])

    CONVERSATIONS.setdefault(req.session_id, []).append(
        Message(role="user", content=raw_user_text, mode=user_mode)
    )

    # Extract potential long-term memories before building the prompt so the
    # freshly captured fact can already inform the answer.
    try:
        autom.maybe_autocapture(req.user_id, req.session_id, raw_user_text)
    except Exception:
        logger.warning("auto-mem capture failed", exc_info=True)

    # build context
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        book_id=req.book_id,
        user_text=raw_user_text,
        mode=user_mode,
        snapshot_override=req.snapshot_override,
    )

    messages: List[Dict[str, str]] = []

    if user_mode == "author":
        # 1) Decide target chapter index (next one)
        book_chapters = [c for c in CHAPTERS.values() if c.book_id == book.id]
        next_index = (max(c.index for c in book_chapters) + 1) if book_chapters else 1

        # 2) Build chapter continuity context (uses previous chapters)
        chap_ctx = build_chapter_context(
            req.user_id,
            book,
            next_index,
            use_prev_chapters=2,
            session_id=req.session_id,
            author_instruction=raw_user_text,
        )

        # 3) Build system prompt from author style pack
        sys = render_system_prompt_author(ctx, book, chap_ctx, req.user_id)
        messages.append({"role": "system", "content": sys})

        # 4) Conversation digest + one-shot instruction (no redundant "Écris un chapitre...")
        history = ctx.short_memory.get("recent_messages", [])
        digest_lines = [f"{m['role']}: {m['content']}" for m in history][-18:]  # TODO use settings.recent_messages_in_context_conversation

        # keep it dead simple: context first, then the user's one-shot instruction
        user_payload = (
            "Contexte de conversation:\n"
            + "\n".join(digest_lines)
            + "\n\nConsignes:\n"
            + raw_user_text.strip()
        ).strip()

        messages.append({"role": "user", "content": user_payload})

        # 5) Call model
        raw_output = call_openai_chat(messages)
        output = raw_output.strip()

        # 6) Persist as the next chapter of this book
        _persist_chapter_from_output(
            user_id=req.user_id,
            book_id=book.id,
            content=output,
            forced_index=next_index,
            used_facets=used_facets,
        )

    else:  # conversation
        register = detect_conv_register(raw_user_text)
        sys = render_system_prompt_conversation(ctx, register, req.user_id)
        messages.append({"role": "system", "content": sys})

        # Inject a few recent turns for continuity
        for m in ctx.short_memory.get("recent_messages", [])[-14:]:
            role = m["role"] if m["role"] in ("user", "assistant") else "user"
            messages.append({"role": role, "content": m["content"]})

        messages.append({"role": "user", "content": raw_user_text})
        raw_output = call_openai_chat(messages)
        output = raw_output.strip()

    # persist assistant message
    CONVERSATIONS[req.session_id].append(
        Message(role="assistant", content=output, mode=user_mode)
    )

    # Refresh rolling summary for long sessions once a full turn is completed
    maybe_refresh_summary(req.session_id)

    # mark facets & memory recency
    snap = SNAPSHOTS.get(req.session_id) or build_snapshot(req.session_id, None)
    snap = mark_facets_used(snap, req.session_id, used_facets)
    SNAPSHOTS[req.session_id] = snap

    rec = MEMORY_USE_RECENCY.setdefault(req.session_id, [])
    now = time.time()
    for mid in used_mem_ids:
        rec.append((mid, now))

    return ChatResponse(output=output, mode=user_mode, used_facets=used_facets, used_memory_ids=used_mem_ids)


@router.post("/api/seed/preferences")
def seed_prefs(req: SeedPrefRequest):
    d = PREFERENCES.setdefault(req.user_id, {})
    deleted, upserted = 0, 0
    for it in req.items:
        val = (it.value or "").strip()
        if val == "":
            if it.key in d:
                del d[it.key]
                deleted += 1
        else:
            d[it.key] = val
            upserted += 1
    return {"ok": True, "upserted": upserted, "deleted": deleted}


@router.get("/api/preferences")
def get_preferences(user_id: str):
    prefs = PREFERENCES.get(user_id, {})
    items = [{"key": k, "value": v} for k, v in prefs.items()]
    return {"ok": True, "items": items}


@router.post("/api/seed/memories")
def seed_memories(req: SeedMemoryRequest):
    lst = MEMORIES.setdefault(req.user_id, [])
    for t in req.texts:
        vec = embed(t)
        lst.append(
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id=req.user_id,
                text=t,
                embedding=vec,
                tags=[],
            )
        )
    return {"ok": True, "count": len(req.texts)}


@router.post("/api/snapshot/set")
def set_snapshot(req: SetSnapshotRequest):
    SNAPSHOTS[req.session_id] = req.snapshot
    return {"ok": True}


@router.post("/api/book/upsert")
def upsert_book(book: Book):
    if not book.id:
        raise HTTPException(status_code=400, detail="book.id is required")
    book.updated_at = time.time()
    BOOKS[book.id] = book
    return {"ok": True, "book_id": book.id}
