import uuid
import time
from fastapi import HTTPException, APIRouter

from app import BOOKS, build_chapter_context, compress_text_for_context, build_context, render_system_prompt_author, render_system_prompt_conversation, ConvRegister, call_openai_chat, \
    summarize_chapter, embed
from models import ChapterEditRequest, ChapterVersion
from stores import CHAPTERS, CHAPTER_VERSIONS, CHAPTER_EMB

router = APIRouter(prefix="/api/chapters", tags=["chapters"])

@router.get("")
def list_chapters(book_id: str):
    items = [c for c in CHAPTERS.values() if c.book_id == book_id]
    items = sorted(items, key=lambda x: x.index)
    return {"ok": True, "items": items}


@router.get("/{chapter_id}")
def get_chapter(chapter_id: str):
    ch = CHAPTERS.get(chapter_id)
    if not ch:
        raise HTTPException(status_code=404, detail="chapter not found")
    return {"ok": True, "item": ch, "versions": CHAPTER_VERSIONS.get(chapter_id, [])}

@router.post("/{chapter_id}/edit")
def edit_chapter(chapter_id: str, req: ChapterEditRequest):
    ch = CHAPTERS.get(chapter_id)
    if not ch:
        raise HTTPException(status_code=404, detail="chapter not found")
    book = BOOKS.get(ch.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    # Build continuity context including the current chapter content
    chap_ctx = build_chapter_context(req.user_id, book, ch.index, use_prev_chapters=2)
    chap_ctx["prev_chapters"].append(f"Chapitre {ch.index} (current) — {ch.title}\n{compress_text_for_context(ch.content)}")

    # Reuse rails
    session_id = ch.book_id
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=session_id,
        user_text=req.edit_instruction,
        mode="rewrite",
        snapshot_override=None,
    )

    # pour les chapitres, on réutilise les rails "conversation" en mode scène
    sys = render_system_prompt_author(ctx, book, chap_ctx, req.user_id)
    sys = sys + "\n\n" + render_system_prompt_conversation(ctx, ConvRegister.scene, req.user_id)

    edit_prompt = (
        "Réécris le chapitre selon les consignes suivantes (français) :\n"
        f"- {req.edit_instruction}\n"
        "- Conserve la continuité, ne change pas les événements clés sauf si demandé.\n"
        "- Garde une dernière ligne percutante; pas de question ni validation.\n"
        "- Conserve/Améliore le titre (sobre).\n\n"
        f"TEXTE ACTUEL:\n{compress_text_for_context(ch.content, 3200)}\n"
    )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": edit_prompt}]
    raw = call_openai_chat(messages)
    out = raw.strip()

    # Versioning first
    prev_ver = ChapterVersion(id=str(uuid.uuid4()), chapter_id=ch.id, title=ch.title, content=ch.content, notes="before-edit")
    CHAPTER_VERSIONS.setdefault(ch.id, []).append(prev_ver)

    # Apply edit
    ch.content = out
    ch.title = (out.splitlines()[0].strip() or ch.title)
    ch.updated_at = time.time()
    ch.summary = summarize_chapter(ch)

    # New version snapshot
    new_ver = ChapterVersion(id=str(uuid.uuid4()), chapter_id=ch.id, parent_version_id=prev_ver.id, title=ch.title, content=ch.content, notes=req.edit_instruction)
    CHAPTER_VERSIONS[ch.id].append(new_ver)

    # Refresh embedding
    try:
        CHAPTER_EMB[ch.id] = embed(ch.content)
    except HTTPException:
        pass

    return {"ok": True, "item": ch, "used_facets": used_facets, "used_memory_ids": used_mem_ids}
