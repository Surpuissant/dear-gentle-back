# app.py
# FastAPI backend for "Dear Gentle — Romance secrète"
# Goals:
# - Strong compliance with the product spec (Henry's persona and modes)
# - Real conversational memory that reduces repetition
# - Strict post-processing to eliminate meta, forbidden endings, emoji overflow
# - Proper handling of modes: conversation vs author vs instruction
# - Europe/Paris time coherence and light space-time hints
# - Clear, practical comments (English), minimal external deps

import os
import re
import time
import uuid
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import auto_memory as autom
import logging
import structlog
from style_packs import (get_style_pack, list_style_meta, render_style_template, STYLE_PACKS)

# --- Configure structlog + stdlib logging
logging.basicConfig(format="%(message)s",level=logging.INFO,)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),processors=[structlog.processors.TimeStamper(fmt="iso"),structlog.processors.JSONRenderer(), ],)
logger = structlog.get_logger("app")
logger.warning("Starting Dear Gentle backend")

# Load environment variables
from dotenv import load_dotenv

from models import Message, Snapshot, InstructionOverride, MemoryItem, ContextPackage, Book, Chapter, ChapterVersion
from utils import paris_now, infer_season_from_date, infer_time_of_day, cosine_sim, build_openai_headers

load_dotenv()

# ----------------------------
# Environment & settings
# ----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt‑4o")
OPENAI_EMB_MODEL = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")
FRONT_ORIGIN = os.getenv("FRONT_ORIGIN", "http://localhost:3000")

if not OPENAI_API_KEY:
    # Fail fast to avoid ambiguous runtime errors
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")


# ----------------------------
# Config knobs (tunable)
# ----------------------------

class Settings(BaseModel):
    # Context window budgeting
    recent_messages_in_context_conversation: int = 20
    recent_messages_in_context_author: int = 40

    # Vector search top-k (long memory)
    emb_top_k_conversation: int = 3
    emb_top_k_author: int = 5

    # Facet scheduling
    facet_cooldown_messages: int = 3
    max_facets_per_output: int = 1

    # Memory reuse cooldown (by last N uses in session)
    cooldown_memory_messages: int = 3

    # MMR control
    mmr_lambda: float = 0.55  # slightly favor relevance

    # How often to refresh rolling summary
    summary_refresh_every_n_turns: int = 5

    # Persona/style rails
    default_style_id: str = "henry"
    forbidden_endings: List[str] = Field(default_factory=lambda: ["bisous"])

    # Conversation register thresholds
    short_words_max: int = 80
    min_scene_punct: int = 6

    # Emoji control
    default_emoji_quota: int = 1


settings = Settings()


class AutoMemActionRequest(BaseModel):
    user_id: str
    ids: List[str] = Field(default_factory=list)

# ----------------------------
# In-memory stores (replace with DB later)
# ----------------------------

from stores import USERS # Structured as Dict[str, Dict[str, str]] = {} -> store minimal user info like preferred style pack etc...
from stores import INSTRUCTIONS # Structured as Dict[str, List[InstructionOverride]] = {}  # user_id -> overrides
CONVERSATIONS: Dict[str, List[Message]] = {}  # session_id -> [Message]
SUMMARIES: Dict[str, str] = {}  # session_id -> rolling summary text
SUMMARY_STATE: Dict[str, Dict[str, Any]] = {}  # session_id -> {"last_turn": int}
PREFERENCES: Dict[str, Dict[str, str]] = {}  # user_id -> {key:value}
MEMORIES: Dict[str, List[MemoryItem]] = {}  # user_id -> [MemoryItem]
SNAPSHOTS: Dict[str, Snapshot] = {}  # session_id -> Snapshot
SNAPSHOT_STATE: Dict[str, Dict[str, Any]] = {}  # session_id -> {overrides, last_generated, last_snapshot}
MEMORY_USE_RECENCY: Dict[str, List[Tuple[str, float]]] = {}  # session_id -> [(mem_id, ts), ...]
BOOKS: Dict[str, Book] = {}

from stores import CHAPTERS # Structured as Dict[str, Chapter] = {} -> store chapters by their ID
from stores import CHAPTER_EMB # Structured as Dict[str, List[float]] = {}  # embedding per chapter content (for later retrieval)
from stores import CHAPTER_VERSIONS # Structured as Dict[str, List[ChapterVersion]] = {}

# Embeddings cache to reduce network calls
_EMB_CACHE: Dict[str, List[float]] = {}


# ----------------------------
# Utility functions
# ----------------------------

def get_current_style_id(user_id: str) -> str:
    sid = USERS.get(user_id, {}).get("style_id")
    return sid or settings.default_style_id

def _effective_forbidden_endings(ctx: ContextPackage, pack) -> list[str]:
    # depuis ctx.instructions (tes overrides actifs déjà calculés) + pack
    base = set(pack.constraints.forbidden_endings or [])
    instr_forb = set(ctx.instructions.get("forbidden_endings") or [])
    return sorted(base.union(instr_forb))


def _format_preferences_block(prefs: Dict[str, str], cap: int = 12) -> str:
    """
    Render user profile as a short, stable block for the system prompt.
    Kept compact to preserve tokens.
    """
    if not prefs:
        return ""
    items = [f"- {k}: {v}" for k, v in prefs.items() if (k and v)]
    if not items:
        return ""
    items = items[:cap]
    header = "Connaissances stables sur l'interlocuteur (profil; à utiliser naturellement, sans poser de questions):"
    return header + "\n" + "\n".join(items)


def compress_text_for_context(text: str, max_tokens: int = 400) -> str:
    """Very light compression using heuristics; replace with a model call if needed."""
    # Keep first 1500 chars and last 800 chars as a naive extract
    if len(text) <= 2300:
        return text
    head = text[:1500]
    tail = text[-800:]
    return head + "\n…\n" + tail


def _clean_line_for_summary(text: str) -> str:
    """Collapse whitespace to keep the summary payload compact."""
    return re.sub(r"\s+", " ", text).strip()


def _format_messages_for_summary(messages: List[Message], limit: int) -> str:
    """Serialize the last ``limit`` conversation messages for the summarizer."""
    window = messages[-limit:]
    lines: List[str] = []
    for m in window:
        if getattr(m, "mode", None) not in (None, "conversation"):
            continue
        content = _clean_line_for_summary(m.content)
        if not content:
            continue
        role = m.role.upper()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def maybe_refresh_summary(session_id: str) -> None:
    """Refresh the rolling summary after a few new conversational turns."""
    conv = CONVERSATIONS.get(session_id, [])
    if not conv:
        return

    # Keep only conversational exchanges (ignore >> instructions and author outputs).
    convo_msgs = [m for m in conv if getattr(m, "mode", None) in (None, "conversation")]
    total_turns = len(convo_msgs)
    if total_turns < 4:
        return

    meta = SUMMARY_STATE.get(session_id, {})
    last_turn = int(meta.get("last_turn", 0))
    if total_turns - last_turn < settings.summary_refresh_every_n_turns and session_id in SUMMARIES:
        return

    window = max(settings.recent_messages_in_context_conversation * 2, 20)
    digest = _format_messages_for_summary(convo_msgs, limit=window)
    if not digest:
        return

    prev_summary = SUMMARIES.get(session_id, "")
    prompt = (
        "Résumé actuel (peut être vide) :\n"
        f"{prev_summary or '(aucun)'}\n\n"
        "Nouvelles répliques à intégrer :\n"
        f"{digest}\n\n"
        "Actualise ce résumé en français (4 à 6 phrases), ton neutre et narratif. "
        "Conserve les faits, émotions et engagements importants. Pas de méta ni de listes."
    )
    messages_payload = [
        {
            "role": "system",
            "content": (
                "Tu résumes une relation épistolaire longue. Mets à jour un résumé compact "
                "en français en 4 à 6 phrases, cohérentes et sans listes à puces."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        summary = call_openai_chat(messages_payload).strip()
    except HTTPException:
        SUMMARY_STATE[session_id] = {"last_turn": total_turns}
        return

    if summary:
        SUMMARIES[session_id] = summary
    SUMMARY_STATE[session_id] = {"last_turn": total_turns}


def summarize_chapter(ch: Chapter) -> str:
    """Create/refresh a compact summary for a chapter using the chat model."""
    sys = "Summarize the chapter in 5-8 sharp bullet points (French). No meta."
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": compress_text_for_context(ch.content, 2800)},
    ]
    try:
        s = call_openai_chat(msgs)
    except HTTPException:
        s = "(résumé indisponible)"
    return s.strip()


def build_chapter_context(
    user_id: str,
    book: Book,
    chapter_index: int,
    use_prev_chapters: int,
    session_id: Optional[str] = None,
    author_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Gather outline beats and a window of previous chapters to enforce continuity."""
    # Collect outline beat for this chapter and neighbors
    outline_beat = book.outline[chapter_index - 1] if 0 < chapter_index <= len(book.outline) else ""
    neighbors = []
    if chapter_index - 2 >= 0 and chapter_index - 2 < len(book.outline):
        neighbors.append((chapter_index - 1, book.outline[chapter_index - 2]))
    if chapter_index < len(book.outline):
        neighbors.append((chapter_index + 1, book.outline[chapter_index]))

    # Previous chapters (compressed)
    prev_ctx: List[str] = []
    target_prev = max(use_prev_chapters, 0)
    chapter_candidates = [
        c for c in CHAPTERS.values() if c.book_id == book.id and c.index < chapter_index
    ]
    chapter_candidates = sorted(chapter_candidates, key=lambda x: x.index)
    sequential_window = chapter_candidates[-target_prev:] if target_prev else []

    summary_hint = SUMMARIES.get(session_id, "") if session_id else ""
    query_parts = [
        outline_beat or "",
        author_instruction or "",
        summary_hint or "",
        " ".join(book.themes or []),
        book.title or "",
    ]
    query_text = "\n".join([part for part in query_parts if part]).strip()
    query_vec: Optional[np.ndarray] = None
    if query_text:
        try:
            vec = np.asarray(embed(query_text), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if np.isfinite(norm) and norm > 1e-9:
                query_vec = vec / norm
        except HTTPException:
            query_vec = None

    if target_prev and chapter_candidates:
        seen_ids: set[str] = set()
        selected_chapters: List[Chapter] = []
        if sequential_window:
            latest = sequential_window[-1]
            selected_chapters.append(latest)
            seen_ids.add(latest.id)

        if query_vec is not None:
            scored: List[Tuple[float, Chapter]] = []
            for ch in chapter_candidates:
                emb = np.asarray(CHAPTER_EMB.get(ch.id, []), dtype=np.float32)
                if emb.ndim != 1 or emb.size == 0 or not np.all(np.isfinite(emb)):
                    continue
                denom = np.linalg.norm(emb)
                if not np.isfinite(denom) or denom <= 1e-9:
                    continue
                emb = emb / denom
                scored.append((float(np.dot(query_vec, emb)), ch))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, ch in scored:
                if ch.id in seen_ids:
                    continue
                selected_chapters.append(ch)
                seen_ids.add(ch.id)
                if len(selected_chapters) >= target_prev:
                    break

        if len(selected_chapters) < target_prev:
            for ch in reversed(sequential_window):
                if ch.id in seen_ids:
                    continue
                selected_chapters.append(ch)
                seen_ids.add(ch.id)
                if len(selected_chapters) >= target_prev:
                    break

        selected_chapters.sort(key=lambda x: x.index)
        for ch in selected_chapters:
            prev_ctx.append(
                f"Chapitre {ch.index} — {ch.title}\n{compress_text_for_context(ch.content)}"
            )

    # User-level long memories that might be relevant to the book
    mems = MEMORIES.get(user_id, [])
    mem_candidates = [m for m in mems if getattr(m, "embedding", None)]
    selected_mems: List[MemoryItem] = []
    if query_vec is not None and mem_candidates:
        k_mem = min(settings.emb_top_k_author, len(mem_candidates))
        selected_mems = mmr_select(
            query_vec=query_vec,
            candidates=mem_candidates,
            k=k_mem,
            lambda_mult=settings.mmr_lambda,
        )
    else:
        selected_mems = mem_candidates[: settings.emb_top_k_author]

    facts = ""
    if selected_mems:
        facts = "\n" + "\n".join(f"- {m.text}" for m in selected_mems)

    return {
        "outline_beat": outline_beat,
        "neighbor_beats": neighbors,
        "prev_chapters": prev_ctx,
        "long_facts": facts,
        "themes": book.themes,
        "style_pref": book.style,
    }


def _persist_chapter_from_output(
        user_id: str,
        book_id: str,
        content: str,
        forced_index: Optional[int] = None,
        used_facets: Optional[List[str]] = None
) -> Chapter:
    """Create Chapter + initial ChapterVersion from an already generated chapter text."""
    book = BOOKS.get(book_id)
    if not book:
        raise HTTPException(status_code=400, detail="book (session_id) not found; call /api/book/upsert first")

    # Determine index
    if forced_index is not None:
        idx = forced_index
    else:
        existing = [c.index for c in CHAPTERS.values() if c.book_id == book_id]
        idx = (max(existing) + 1) if existing else 1

    # Title from first line (fallback)
    title = (content.splitlines()[0].strip() or f"Chapitre {idx}")

    ch = Chapter(
        id=str(uuid.uuid4()),
        book_id=book_id,
        index=idx,
        title=title,
        content=content,
        model=OPENAI_CHAT_MODEL,
    )
    ch.summary = summarize_chapter(ch)
    CHAPTERS[ch.id] = ch

    ver = ChapterVersion(id=str(uuid.uuid4()), chapter_id=ch.id, title=ch.title, content=ch.content, notes="v1-from-chat")
    CHAPTER_VERSIONS.setdefault(ch.id, []).append(ver)

    # Embed for retrieval (best effort)
    try:
        CHAPTER_EMB[ch.id] = embed(ch.content)
    except HTTPException:
        CHAPTER_EMB[ch.id] = []

    # Mark facet usage
    session_id = book_id
    snap = SNAPSHOTS.get(session_id) or build_snapshot(session_id, None)
    if used_facets:
        snap = mark_facets_used(snap, session_id, used_facets)
        SNAPSHOTS[session_id] = snap

    return ch


# ----------------------------
# Chapter editing helper
# ----------------------------

def perform_chapter_edit(
    chapter: Chapter,
    user_id: str,
    edit_instruction: str,
) -> Tuple[Chapter, List[str], List[str]]:
    """Shared logic to rewrite an existing chapter in place."""

    book = BOOKS.get(chapter.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    # Build continuity context including the current chapter content
    chap_ctx = build_chapter_context(
        user_id,
        book,
        chapter.index,
        use_prev_chapters=2,
        session_id=chapter.book_id,
        author_instruction=edit_instruction,
    )
    chap_ctx["prev_chapters"].append(
        f"Chapitre {chapter.index} (current) — {chapter.title}\n{compress_text_for_context(chapter.content)}"
    )

    ctx, used_facets, used_mem_ids = build_context(
        user_id=user_id,
        session_id=chapter.book_id,
        book_id=book.id,
        user_text=edit_instruction,
        mode="rewrite",
        snapshot_override=None,
    )

    sys = render_system_prompt_author(ctx, book, chap_ctx, user_id)

    edit_prompt = (
        "Réécris le chapitre selon les consignes suivantes (français) :\n"
        f"- {edit_instruction}\n"
        "- Conserve la continuité, ne change pas les événements clés sauf si demandé.\n"
        "- Garde une dernière ligne percutante; pas de question ni validation.\n"
        "- Conserve/Améliore le titre (sobre).\n\n"
        f"TEXTE ACTUEL:\n{compress_text_for_context(chapter.content, 3200)}\n"
    )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": edit_prompt}]
    raw = call_openai_chat(messages)
    out = raw.strip()

    prev_ver = ChapterVersion(
        id=str(uuid.uuid4()),
        chapter_id=chapter.id,
        title=chapter.title,
        content=chapter.content,
        notes="before-edit",
    )
    CHAPTER_VERSIONS.setdefault(chapter.id, []).append(prev_ver)

    chapter.content = out
    chapter.title = (out.splitlines()[0].strip() or chapter.title)
    chapter.updated_at = time.time()
    chapter.summary = summarize_chapter(chapter)

    new_ver = ChapterVersion(
        id=str(uuid.uuid4()),
        chapter_id=chapter.id,
        parent_version_id=prev_ver.id,
        title=chapter.title,
        content=chapter.content,
        notes=edit_instruction,
    )
    CHAPTER_VERSIONS[chapter.id].append(new_ver)

    try:
        CHAPTER_EMB[chapter.id] = embed(chapter.content)
    except HTTPException:
        pass

    return chapter, used_facets, used_mem_ids


# ----------------------------
# Mode detection
# ----------------------------

def detect_mode(user_text: str, default_mode: str = "conversation") -> str:
    """Return one of: 'author' | 'conversation' | 'instruction'.
    Conventions:
    - lines starting with '>>' => instruction
    - lines starting with '::author' or '::a' => author (strip marker)
    - otherwise => conversation
    """
    t = user_text.strip()
    if t.startswith(">>"):
        return "instruction"
    if t.lower().startswith("::author") or t.lower().startswith("::a"):
        return "author"
    return default_mode


def strip_mode_marker(text: str) -> str:
    t = text.strip()
    if t.startswith(">>"):
        return t  # handled elsewhere
    if t.lower().startswith("::author"):
        return t[len("::author"):].lstrip()
    if t.lower().startswith("::a"):
        return t[len("::a"):].lstrip()
    return text

# ----------------------------
# Conversation register detection
# ----------------------------

class ConvRegister(str):
    brevity = "brevity"  # short, poetic lines
    scene = "scene"  # narrative + dialogue


def detect_conv_register(user_text: str) -> str:
    """Heuristic to choose Henry's register.
    - brevity if short and not narration-heavy
    - scene   if the user starts with a narrative passage (long, punctuated, not a question)
    """
    t = user_text.strip()
    # obvious narrative cues: multiple sentences, commas/ellipses, first-person past/present
    words = len(t.split())
    punct = sum(t.count(p) for p in [".", ",", ";", "—", "…", "!"])
    if words > settings.short_words_max and punct >= settings.min_scene_punct and not t.endswith("?"):
        return ConvRegister.scene
    # if user begins with dialogue or scene markers
    if re.match(r'^[«\"\-\(].{10,}', t):
        return ConvRegister.scene
    return ConvRegister.brevity


# ----------------------------
# Facet scheduler (place/time/weather/context)
# ----------------------------

def _conversation_only_history(session_id: str) -> List[Message]:
    """Return session history limited to conversational exchanges."""
    return [
        m
        for m in CONVERSATIONS.get(session_id, [])
        if getattr(m, "mode", None) in (None, "conversation")
        and m.role in {"user", "assistant"}
    ]


def schedule_facets(snapshot: Snapshot, session_id: str) -> List[str]:
    """
    Choose up to max_facets_per_output facets (with cooldown).
    We store "last_mentioned_facets" as message-index markers.
    """
    candidates = []

    if snapshot.time_of_day:
        candidates.append(("time_of_day", snapshot.time_of_day))
    if snapshot.season:
        candidates.append(("season", snapshot.season))
    if snapshot.weather and snapshot.weather.get("condition"):
        candidates.append(("weather", snapshot.weather["condition"]))
    for cf in snapshot.contextual_facets:
        candidates.append(("context", cf))

    # Cooldown check
    last_map: Dict[str, int] = {}
    for rec in snapshot.last_mentioned_facets or []:
        facet = rec.get("facet")
        ts_raw = rec.get("ts")
        if not facet:
            continue
        try:
            last_map[facet] = int(ts_raw) if ts_raw is not None else 0
        except (TypeError, ValueError):
            continue

    allowed = []
    recent_msgs = len(_conversation_only_history(session_id))
    for kind, facet in candidates:
        last_ts = last_map.get(facet)
        if last_ts is None:
            allowed.append(facet)
            continue
        if (recent_msgs - last_ts) >= settings.facet_cooldown_messages:
            allowed.append(facet)

    return allowed[: settings.max_facets_per_output]


def mark_facets_used(snapshot: Snapshot, session_id: str, used: List[str]) -> Snapshot:
    """Record facet usage as message-index pseudo-timestamps."""
    if not used:
        return snapshot
    current_idx = len(_conversation_only_history(session_id))
    records = snapshot.last_mentioned_facets or []
    for f in used:
        records = [r for r in records if r.get("facet") != f]
        records.append({"facet": f, "ts": str(current_idx)})
    snapshot.last_mentioned_facets = records
    state = _snapshot_state(session_id)
    state["last_snapshot"] = snapshot
    SNAPSHOTS[session_id] = snapshot
    return snapshot


# ----------------------------
# Embeddings + MMR selection
# ----------------------------

def embed(text: str, retries: int = 2, timeout_s: int = 30) -> List[float]:
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    url = f"{OPENAI_BASE_URL}/embeddings"
    payload = {"model": OPENAI_EMB_MODEL, "input": text}
    headers = build_openai_headers(OPENAI_API_KEY)

    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            if r.status_code == 200:
                vec = r.json()["data"][0]["embedding"]
                if not vec:  # hard guard
                    raise HTTPException(status_code=502, detail="Empty embedding from provider")
                _EMB_CACHE[text] = vec
                return vec
            last_err = {"status": r.status_code, "body": r.text[:2000]}
        except requests.RequestException as e:
            last_err = {"exception": str(e)}
        time.sleep(0.8 * (2 ** i))
    logger.error("OpenAI embeddings error after retries", last_err=last_err, model=OPENAI_EMB_MODEL)
    raise HTTPException(status_code=502, detail={"where": "embeddings", "last_err": last_err})



def mmr_select(query_vec: np.ndarray, candidates: List[MemoryItem], k: int, lambda_mult: float,used_recent_ids: Optional[List[str]] = None) -> List[MemoryItem]:
    if k <= 0 or not candidates or query_vec is None:
        return []

    used_recent_ids = set(used_recent_ids or [])

    # --- sanitize & normalize candidates
    valid_idx, valid_items, cand_vecs = [], [], []
    dim = None
    for i, m in enumerate(candidates):
        if m.id in used_recent_ids:
            logger.info("memory_skipped_recent", memory_id=m.id)
            continue
        v = np.asarray(getattr(m, "embedding", []), dtype=np.float32)
        if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
            continue
        if dim is None:
            dim = v.size
        if v.size != dim:
            continue
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 1e-12:
            continue
        valid_idx.append(i)
        valid_items.append(m)
        cand_vecs.append(v / n)

    if not valid_items:
        return []

    # --- sanitize & normalize query
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if q.size != cand_vecs[0].size or not np.all(np.isfinite(q)):
        return []
    qn = np.linalg.norm(q)
    if not np.isfinite(qn) or qn <= 1e-12:
        return []
    q = q / qn

    lam = float(np.clip(lambda_mult, 0.0, 1.0))  # keep MMR bounds
    q_sims = np.array([float(np.dot(q, v)) for v in cand_vecs], dtype=np.float32)

    selected_local: List[int] = []
    target = min(k, len(valid_items))
    while len(selected_local) < target:
        best_i, best_score = -1, -1e9
        for i in range(len(valid_items)):
            if i in selected_local:
                continue
            if selected_local:
                # cosine == dot because vectors are unit-norm
                max_sim = max(float(np.dot(cand_vecs[i], cand_vecs[j])) for j in selected_local)
            else:
                max_sim = 0.0
            score = lam * q_sims[i] - (1.0 - lam) * max_sim
            # deterministic tie-break: plus petit index
            if score > best_score or (score == best_score and i < best_i):
                best_i, best_score = i, score
        if best_i == -1:
            break
        selected_local.append(best_i)

    return [valid_items[i] for i in selected_local]


# ----------------------------
# Snapshot builder
# ----------------------------

SNAPSHOT_REFRESH_SECONDS = 15 * 60  # 15 minutes


def _snapshot_state(session_id: str) -> Dict[str, Any]:
    return SNAPSHOT_STATE.setdefault(
        session_id,
        {"overrides": None, "last_generated": None, "last_snapshot": None},
    )


def _default_weather_for_time(time_of_day: Optional[str]) -> Dict[str, str]:
    mapping = {
        "morning": "fresh_morning",
        "afternoon": "soft_afternoon",
        "evening": "mild_evening",
        "night": "quiet_night",
    }
    return {"condition": mapping.get(time_of_day or "", "mild_evening")}


def _default_contextual_facets(time_of_day: Optional[str]) -> List[str]:
    base = {
        "morning": ["terrasse ensoleillée au bord du lac"],
        "afternoon": ["promenade le long du canal du Thiou"],
        "evening": ["terrasse au bord du lac"],
        "night": ["appartement cosy avec vue sur le lac"],
    }
    return base.get(time_of_day, ["terrasse au bord du lac"]).copy()


def build_snapshot(session_id: str, override: Optional[Snapshot]) -> Snapshot:
    """Create/merge the snapshot and choose exactly one facet for this turn."""
    state = _snapshot_state(session_id)

    stored_override: Optional[Snapshot] = state.get("overrides")
    if override:
        data = stored_override.dict(exclude_none=True) if stored_override else {}
        data.update(override.dict(exclude_none=True))
        stored_override = Snapshot(**data)
        state["overrides"] = stored_override

    now = paris_now()
    last_generated = state.get("last_generated")
    refresh_dynamic = True
    if last_generated is not None:
        try:
            delta = (now - last_generated).total_seconds()
        except TypeError:
            # Fallback in case legacy state stored as str
            delta = SNAPSHOT_REFRESH_SECONDS + 1
        refresh_dynamic = delta >= SNAPSHOT_REFRESH_SECONDS

    previous: Optional[Snapshot] = state.get("last_snapshot")

    base_time_of_day = infer_time_of_day(now)
    base_season = infer_season_from_date(now)

    if refresh_dynamic or not previous:
        time_of_day = base_time_of_day
        season = base_season
        weather = _default_weather_for_time(time_of_day)
        contextual_facets = _default_contextual_facets(time_of_day)
    else:
        time_of_day = previous.time_of_day
        season = previous.season
        weather = previous.weather or _default_weather_for_time(previous.time_of_day)
        contextual_facets = (previous.contextual_facets or [])[:]

    last_mentioned = (previous.last_mentioned_facets or [])[:] if previous else []
    cultural_refs = (previous.cultural_refs or [])[:] if previous else []

    snap = Snapshot(
        location={"city": "Annecy", "country": "France"},
        datetime_local_iso=now.isoformat(),
        season=season,
        time_of_day=time_of_day,
        weather=weather,
        contextual_facets=contextual_facets,
        last_mentioned_facets=last_mentioned,
        cultural_refs=cultural_refs,
    )

    if stored_override:
        data = snap.dict()
        for k, v in stored_override.dict(exclude_none=True).items():
            data[k] = v
        snap = Snapshot(**data)

    chosen = schedule_facets(snap, session_id)
    snap.selected_facet = chosen[0] if chosen else None

    state["last_generated"] = now
    state["last_snapshot"] = snap
    SNAPSHOTS[session_id] = snap

    return snap


def apply_snapshot_override(session_id: str, snapshot: Snapshot) -> Snapshot:
    """Persist user overrides then rebuild a fresh snapshot."""
    state = _snapshot_state(session_id)
    stored = state.get("overrides")
    data = stored.dict(exclude_none=True) if isinstance(stored, Snapshot) else {}
    data.update(snapshot.dict(exclude_none=True))
    state["overrides"] = Snapshot(**data)
    state["last_snapshot"] = None
    state["last_generated"] = None
    return build_snapshot(session_id, None)


# ----------------------------
# Context builder
# ----------------------------


def build_context(
    user_id: str,
    session_id: str,
    book_id: str,
    user_text: str,
    mode: str,
    snapshot_override: Optional[Snapshot],
) -> Tuple[ContextPackage, List[str], List[str]]:
    """Assemble runtime context for Henry.

    Historically this function tried to recover the ``Book`` using the
    ``session_id`` which is unrelated.  That subtle mismatch meant that
    preferences such as the selected writing style were silently dropped
    (``book`` resolved to ``None``) and the system prompt lost its guard‑rails.
    We now take the ``book_id`` explicitly to make the dependency obvious.
    """

    book = BOOKS.get(book_id)
    style_pack = get_style_pack(get_current_style_id(user_id))
    prefs = PREFERENCES.get(user_id, {})

    active_instr = [i for i in INSTRUCTIONS.get(user_id, []) if i.active]

    # Base immuable (snapshot de démarrage)
    base_forbidden = list(settings.forbidden_endings)

    # Merge overrides actifs
    merged_forbidden = list(base_forbidden)
    base_lc = [x.lower() for x in base_forbidden]
    for ov in active_instr:
        if getattr(ov, "rule_key", "") == "forbidden_ending_add":
            val = (ov.rule_value or "").strip()
            if val and val.lower() not in base_lc and val.lower() not in [x.lower() for x in merged_forbidden]:
                merged_forbidden.append(val)

    instructions = {
        # When a book is missing (should be rare) we gracefully fall back to the
        # default rails so the assistant keeps behaving.
        "style": getattr(book, "style", None),
        "forbidden_endings": merged_forbidden,
        "bounds": ["no_vulgarity", "no_intrusive_secrets"],
        "preferences": prefs,
        "raw_instructions": [i.rule_value for i in active_instr if getattr(i, "rule_key", "") == "raw"],
    }

    snapshot = build_snapshot(session_id, snapshot_override)
    used_facets = [snapshot.selected_facet] if snapshot.selected_facet else []

    msgs = CONVERSATIONS.get(session_id, [])

    # Keep only true conversational exchanges for runtime context.
    conversation_msgs = [
        m
        for m in msgs
        if getattr(m, "mode", None) in (None, "conversation")
        and m.role in {"user", "assistant"}
    ]

    authorial_mode = mode in {"author", "rewrite"}
    if authorial_mode:
        # Chapter writing and rewrites benefit from a broader context window and
        # richer long-term memory selection.
        k = settings.recent_messages_in_context_author
        top_k = settings.emb_top_k_author
    else:  # conversation
        k = settings.recent_messages_in_context_conversation
        top_k = settings.emb_top_k_conversation

    # Derniers messages (exclusivement conversationnels)
    recent_msgs = [
        {"role": m.role, "content": m.content} for m in conversation_msgs[-k:]
    ]

    short_memory = {
        "recent_messages": recent_msgs,
    }

    is_first_turn = False
    if not authorial_mode:
        is_first_turn = not any(m.role == "assistant" for m in conversation_msgs)

    rolling_summary = SUMMARIES.get(session_id)
    if rolling_summary:
        short_memory["rolling_summary"] = rolling_summary

    long_list = MEMORIES.get(user_id, [])

    used_recent_ids = [mid for mid, _ in MEMORY_USE_RECENCY.get(session_id, [])[-settings.cooldown_memory_messages:]]

    selected_memories: List[MemoryItem] = []
    if top_k > 0 and long_list:
        try:
            # Encode query for retrieval
            q_vec = np.array(embed(user_text), dtype=float)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
            # filtre candidats invalides
            clean_cands = [m for m in long_list if getattr(m, "embedding", None) and len(m.embedding) > 0]
            selected_memories = mmr_select(
                query_vec=q_vec,
                candidates=clean_cands,
                k=top_k,
                lambda_mult=settings.mmr_lambda,
                used_recent_ids=used_recent_ids,
            )
        except HTTPException:
            selected_memories = []
    else:
        q_vec = None  # pas d’embed inutile

    ctx = ContextPackage(
        mode=mode,
        instructions=instructions,
        snapshot=snapshot,
        short_memory=short_memory,
        long_memory={"facts": [{"id": m.id, "text": m.text} for m in selected_memories]},
        initial_situation=style_pack.initial_situation if mode == "conversation" else None,
        is_first_turn=is_first_turn,
    )
    used_mem_ids = [m.id for m in selected_memories]
    return ctx, used_facets, used_mem_ids


# ----------------------------
# System prompt renderer
# ----------------------------

# This is where we assemble the final system prompt from the style pack template + context pieces
def _build_common_ctx(ctx: ContextPackage, register: Optional[str]) -> dict:
    snap = ctx.snapshot
    instr = ctx.instructions

    # prefs
    prefs_block = _format_preferences_block(instr.get("preferences", {}))
    # raw instructions
    raw_instr = instr.get("raw_instructions") or []
    raw_instructions = ""
    if raw_instr:
        raw_instructions = "Consignes de l'utilisateur:\n" + "\n".join(f"- {it}" for it in raw_instr[:8])

    summary_block = ""
    rolling_summary = ctx.short_memory.get("rolling_summary") if ctx.short_memory else None
    if rolling_summary:
        summary_block = "Résumé condensé des échanges précédents:\n" + rolling_summary

    space_time_hint = f"{snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather} | facet:{snap.selected_facet}"

    initial_situation_block = ""
    initial_situation_value = ""
    if getattr(ctx, "is_first_turn", False) and ctx.initial_situation:
        initial_situation_block = (
            "Situation initiale pour ce premier échange :\n"
            f"{ctx.initial_situation}\n\n"
        )
        initial_situation_value = ctx.initial_situation

    # constraints will be filled from style
    return {
        "prefs_block": prefs_block,
        "raw_instructions": raw_instructions,
        "summary_block": summary_block,
        "space_time_hint": space_time_hint,
        "register": register or "",
        "initial_situation_block": initial_situation_block,
        "initial_situation": initial_situation_value,
        "is_first_turn": getattr(ctx, "is_first_turn", False),
    }

def _render_output_rails_from_style(pack, ctx: ContextPackage, end_with_statement: bool) -> str:
    forbidden = ", ".join(_effective_forbidden_endings(ctx, pack)) or "—"
    endings_rule = "- Termine par une phrase déclarative (pas de question)." if end_with_statement else ""
    return render_style_template(
        pack.templates.output_rails,
        {
            "emoji_quota": pack.constraints.emoji_quota,
            "forbidden_endings": forbidden,
            "endings_rule": endings_rule,
        }
    )


def _render_output_rails(ctx: ContextPackage, register: Optional[str]) -> str:
    forb = ctx.instructions.get("forbidden_endings", []) or []
    emoji_quota = settings.default_emoji_quota  # tu peux le garder simple
    lines = [
        "Règles de sortie (respect strict) :",
        "- Aucune méta‑intro ni méta‑conclusion.",
        f"- Émojis : au plus {emoji_quota} dans tout le message.",
        f"- Interdictions de fin : {', '.join(forb) if forb else '—'}.",
    ]
    if ctx.mode == "conversation" and register == ConvRegister.scene:
        lines.append("- Termine par une phrase déclarative (pas de question).")
    return "\n".join(lines)


# app.py
def render_system_prompt_conversation(ctx: ContextPackage, register: str, user_id: str) -> str:
    pack = get_style_pack(get_current_style_id(user_id))
    base = _build_common_ctx(ctx, register)
    output_rails = _render_output_rails_from_style(pack, ctx, end_with_statement=(ctx.mode=="conversation" and register==ConvRegister.scene))
    common = {**base, "output_rails": output_rails}
    tpl = pack.templates.conversation_brevity if register == ConvRegister.brevity else pack.templates.conversation_scene
    return render_style_template(tpl, common)

def render_system_prompt_author(ctx: ContextPackage, book: Book, chap_ctx: Dict[str, Any], user_id: str) -> str:
    # Use user-selected style (fallback to default)
    pack = get_style_pack(get_current_style_id(user_id))

    base = _build_common_ctx(ctx, register=None)

    # Chapter context (neighbors & previous)
    prev = "\n---\n".join(chap_ctx.get("prev_chapters", []))
    neighbor_beats = "; ".join([f"N{idx}:{beat}" for idx, beat in chap_ctx.get("neighbor_beats", [])])

    # Style rails (forbidden_endings + emoji_quota, merged with runtime overrides)
    output_rails = _render_output_rails_from_style(pack, ctx, end_with_statement=False)

    render_ctx = {
        **base,
        "book_title": book.title,
        "themes": ", ".join(chap_ctx.get("themes", [])),
        "style": chap_ctx.get("style_pref", book.style),
        "outline_beat": chap_ctx.get("outline_beat", ""),
        "neighbor_beats": neighbor_beats,
        "prev_chapters": prev,
        "long_facts": chap_ctx.get("long_facts", "") or "—",
        "output_rails": output_rails,
    }

    return render_style_template(pack.templates.author, render_ctx)

# ----------------------------
# OpenAI call (robust)
# ----------------------------

def call_openai_chat(
    messages: List[Dict[str, str]],
    retries: int = 2,
    *,
    temperature: float = 0.9,
    presence_penalty: float = 0.4,
    frequency_penalty: float = 0.4,
) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    headers = build_openai_headers(OPENAI_API_KEY)

    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            # capture corps d'erreur lisible
            try:
                err_json = r.json()
            except Exception:
                err_json = {"raw_text": r.text}
            last_err = {"status": r.status_code, "error": err_json}
        except requests.RequestException as e:
            last_err = {"exception": str(e)}
        time.sleep(0.8 * (2 ** i))

    # -> Log lisible
    logger.error(
        "OpenAI chat error after retries",
        extra={
            "last_err": last_err,
            "model": OPENAI_CHAT_MODEL,
            "base_url": OPENAI_BASE_URL,
        },
    )
    # -> Et remonte un body JSON exploitable côté frontend/curl
    raise HTTPException(
        status_code=502,
        detail={
            "where": "chat",
            "model": OPENAI_CHAT_MODEL,
            "base_url": OPENAI_BASE_URL,
            "last_err": last_err,
        },
    )


# ----------------------------
# Auto-memory integration
# ----------------------------

autom.embed = embed
autom.cosine_sim = cosine_sim
autom.call_openai_chat = call_openai_chat
autom.MEMORIES = MEMORIES
autom.MEMORY_USE_RECENCY = MEMORY_USE_RECENCY

# ----------------------------
# FastAPI app
# ----------------------------
from routes.styles import router as styles_router
from routes.instructions import router as instructions_router
from routes.chapters import router as chapters_router
from routes.health import router as health_router
from routes.core import router as core_router
from routes.docs import router as docs_router
app = FastAPI(title="Dear Gentle — Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register styles routes
app.include_router(styles_router)
app.include_router(instructions_router)
app.include_router(chapters_router)
app.include_router(health_router)
app.include_router(docs_router)
app.include_router(core_router)

