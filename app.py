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

from models import Message, Snapshot, InstructionOverride, MemoryItem, ContextPackage, ChatRequest, ChatResponse, SeedPrefRequest, SeedMemoryRequest, SetSnapshotRequest, Book, Chapter, ChapterVersion, ChapterGenRequest, ChapterEditRequest
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
PREFERENCES: Dict[str, Dict[str, str]] = {}  # user_id -> {key:value}
MEMORIES: Dict[str, List[MemoryItem]] = {}  # user_id -> [MemoryItem]
SNAPSHOTS: Dict[str, Snapshot] = {}  # session_id -> Snapshot
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


def build_chapter_context(user_id: str, book: Book, chapter_index: int, use_prev_chapters: int) -> Dict[str, Any]:
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
    if use_prev_chapters > 0:
        prev = [c for c in CHAPTERS.values() if c.book_id == book.id and c.index < chapter_index]
        prev = sorted(prev, key=lambda x: x.index)[-use_prev_chapters:]
        for pc in prev:
            prev_ctx.append(f"Chapitre {pc.index} — {pc.title}\n{compress_text_for_context(pc.content)}")

    # User-level long memories that might be relevant to the book
    mems = MEMORIES.get(user_id, [])
    facts = "; ".join([m.text for m in mems[:5]])  # naive cap, could MMR on a synthetic query

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
    last_map = {x["facet"]: x["ts"] for x in (snapshot.last_mentioned_facets or [])}
    allowed = []
    recent_msgs = len(CONVERSATIONS.get(session_id, []))
    for kind, facet in candidates:
        last_ts = last_map.get(facet)
        if last_ts is None:
            allowed.append(facet)
            continue
        try:
            last_idx = int(last_ts)
        except:
            last_idx = 0
        if (recent_msgs - last_idx) >= settings.facet_cooldown_messages:
            allowed.append(facet)

    return allowed[: settings.max_facets_per_output]


def mark_facets_used(snapshot: Snapshot, session_id: str, used: List[str]) -> Snapshot:
    """Record facet usage as message-index pseudo-timestamps."""
    if not used:
        return snapshot
    current_idx = len(CONVERSATIONS.get(session_id, []))
    records = snapshot.last_mentioned_facets or []
    for f in used:
        records = [r for r in records if r.get("facet") != f]
        records.append({"facet": f, "ts": str(current_idx)})
    snapshot.last_mentioned_facets = records
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

def build_snapshot(session_id: str, override: Optional[Snapshot]) -> Snapshot:
    """Create/merge the snapshot and choose exactly one facet for this turn."""
    snap = SNAPSHOTS.get(session_id)
    if not snap:
        d = paris_now()
        snap = Snapshot(
            location={"city": "Annecy", "country": "France"},
            datetime_local_iso=d.isoformat(),
            season=infer_season_from_date(d),
            time_of_day=infer_time_of_day(d),
            weather={"condition": "mild_evening"},
            contextual_facets=["terrasse au bord du lac"],
            cultural_refs=[]
        )

    if override:
        # Shallow merge; override takes precedence if fields are present
        data = snap.dict()
        for k, v in override.dict(exclude_none=True).items():
            data[k] = v
        snap = Snapshot(**data)

    chosen = schedule_facets(snap, session_id)
    snap.selected_facet = chosen[0] if chosen else None
    return snap


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

    # Filtrer les ">>" du contexte
    msgs_no_instr = [m for m in msgs if getattr(m, "mode", None) != "instruction"]

    authorial_mode = mode in {"author", "rewrite"}
    if authorial_mode:
        # Chapter writing and rewrites benefit from a broader context window and
        # richer long-term memory selection.
        k = settings.recent_messages_in_context_author
        top_k = settings.emb_top_k_author
    else:  # conversation
        k = settings.recent_messages_in_context_conversation
        top_k = settings.emb_top_k_conversation

    # Derniers messages (sans instructions)
    recent_msgs = [{"role": m.role, "content": m.content} for m in msgs_no_instr[-k:]]

    short_memory = {
        "recent_messages": recent_msgs,
    }

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

    space_time_hint = f"{snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather} | facet:{snap.selected_facet}"

    # constraints will be filled from style
    return {
        "prefs_block": prefs_block,
        "raw_instructions": raw_instructions,
        "space_time_hint": space_time_hint,
        "register": register or "",
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
    pack = get_style_pack(PREFERENCES.get(user_id, {}).get("style_id") or settings.default_style_id)
    base = _build_common_ctx(ctx, register)
    output_rails = _render_output_rails_from_style(pack, ctx, end_with_statement=(ctx.mode=="conversation" and register==ConvRegister.scene))
    common = {**base, "output_rails": output_rails}
    tpl = pack.templates.conversation_brevity if register == ConvRegister.brevity else pack.templates.conversation_scene
    return render_style_template(tpl, common)

def render_system_prompt_author(ctx: ContextPackage, book: Book, chap_ctx: Dict[str, Any], user_id: str) -> str:
    # Use user-selected style (fallback to default)
    pack = get_style_pack(PREFERENCES.get(user_id, {}).get("style_id") or settings.default_style_id)

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

def call_openai_chat(messages: List[Dict[str, str]], retries: int = 2) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.4,
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

# ----------------------------
# HTTP endpoints
# ----------------------------


@app.get("/api/memories/auto/pending")
def get_auto_pending(user_id: str):
    items = autom.list_pending(user_id)
    logger.info("auto_mem_pending_list", user_id=user_id, count=len(items))
    return {"ok": True, "items": [c.dict() for c in items]}


@app.post("/api/memories/auto/accept")
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


@app.post("/api/memories/auto/reject")
def post_auto_reject(req: AutoMemActionRequest):
    result = autom.reject_pending(req.user_id, req.ids)
    logger.info(
        "auto_mem_pending_reject",
        user_id=req.user_id,
        requested_ids=req.ids,
        deleted=result.get("deleted", 0),
    )
    return result


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    book = BOOKS.get(req.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    user_mode = detect_mode(req.message, default_mode="conversation")
    raw_user_text = strip_mode_marker(req.message)

    # instruction mode: store preference/memory, append message, return empty output
    if user_mode == "instruction":
        # 1) store raw instruction (verbatim) in INSTRUCTIONS
        instr_list = INSTRUCTIONS.setdefault(req.user_id, [])
        instr_list.append(InstructionOverride(
            rule_key="raw",
            rule_value=req.message[2:].strip(),  # strip the leading '>>'
            active=True
        ))
        # 2) garder une trace dans CONVERSATIONS pour l’audit (tu filtres déjà ces messages du contexte)
        CONVERSATIONS.setdefault(req.session_id, []).append(
            Message(role="user", content=req.message, mode=user_mode)
        )

        # 3) important: ne pas appeler embed(), ne pas toucher MEMORIES
        return ChatResponse(output="", mode="instruction", used_facets=[], used_memory_ids=[])

    # persist user message
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
        chap_ctx = build_chapter_context(req.user_id, book, next_index, use_prev_chapters=2)

        # 3) Build system prompt from author style pack
        sys = render_system_prompt_author(ctx, book, chap_ctx, req.user_id)
        messages.append({"role": "system", "content": sys})

        # 4) Conversation digest + one-shot instruction (no redundant "Écris un chapitre...")
        history = ctx.short_memory.get("recent_messages", [])
        digest_lines = [f"{m['role']}: {m['content']}" for m in history][-18:] # TODO use settings.recent_messages_in_context_conversation

        # keep it dead simple: context first, then the user's one-shot instruction
        user_payload = (
                "Contexte de conversation:\n" +
                "\n".join(digest_lines) +
                "\n\nConsignes:\n" +
                raw_user_text.strip()
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
    CONVERSATIONS[req.session_id].append(Message(role="assistant", content=output, mode=user_mode))

    # mark facets & memory recency
    snap = SNAPSHOTS.get(req.session_id) or build_snapshot(req.session_id, None)
    snap = mark_facets_used(snap, req.session_id, used_facets)
    SNAPSHOTS[req.session_id] = snap

    rec = MEMORY_USE_RECENCY.setdefault(req.session_id, [])
    now = time.time()
    for mid in used_mem_ids:
        rec.append((mid, now))

    return ChatResponse(output=output, mode=user_mode, used_facets=used_facets, used_memory_ids=used_mem_ids)


@app.post("/api/seed/preferences")
def seed_prefs(req: SeedPrefRequest):
    """
    Upsert user preferences. If a value is the empty string, delete the key.
    """
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


@app.get("/api/preferences")
def get_preferences(user_id: str):
    """Return all stored preferences for a user as a list of {key,value}."""
    prefs = PREFERENCES.get(user_id, {})
    items = [{"key": k, "value": v} for k, v in prefs.items()]
    return {"ok": True, "items": items}


@app.post("/api/seed/memories")
def seed_memories(req: SeedMemoryRequest):
    """
    Seed long-term memories for a user:
    - Each text is embedded once and stored with its vector.
    - In production, replace this with DB persistence.
    """
    lst = MEMORIES.setdefault(req.user_id, [])
    for t in req.texts:
        vec = embed(t)
        lst.append(MemoryItem(
            id=str(uuid.uuid4()),
            user_id=req.user_id,
            text=t,
            embedding=vec,
            tags=[]
        ))
    return {"ok": True, "count": len(req.texts)}


@app.post("/api/snapshot/set")
def set_snapshot(req: SetSnapshotRequest):
    """Allow the front-end to set/override the world-state snapshot."""
    SNAPSHOTS[req.session_id] = req.snapshot
    return {"ok": True}


@app.post("/api/book/upsert")
def upsert_book(book: Book):
    """Create or update a Book (id provided by client to keep things simple)."""
    if not book.id:
        raise HTTPException(status_code=400, detail="book.id is required")
    book.updated_at = time.time()
    BOOKS[book.id] = book
    return {"ok": True, "book_id": book.id}


