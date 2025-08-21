# app.py
# FastAPI backend for "Dear Gentle — Romance secrète"
# Goals:
# - Strong compliance with the product spec (Henry's persona and modes)
# - Real conversational memory that reduces repetition:
#   * Short-memory: recent turns + compact rolling summary
#   * Long-memory: vector search (OpenAI embeddings) with MMR + reuse cooldown
# - Strict post-processing to eliminate meta, forbidden endings, emoji overflow
# - Proper handling of modes: short, chapter, rewrite (++), instruction (>>)
# - Europe/Paris time coherence and light space-time hints
# - Clear, practical comments (English), minimal external deps

import os
import re
import time
import uuid
from typing import List, Dict, Optional, Tuple, Any
import re as _re
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("app.openai")

# Load environment variables
from dotenv import load_dotenv

from models import Message, Snapshot, InstructionOverride, MemoryItem, ContextPackage, ChatRequest, ChatResponse, SeedPrefRequest, SeedMemoryRequest, SetSnapshotRequest, Book, Chapter, ChapterVersion, \
    ChapterGenRequest, ChapterEditRequest
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
    mmr_lambda: float = 0.55 # slightly favor relevance

    # How often to refresh rolling summary
    summary_refresh_every_n_turns: int = 5

    # Persona/style rails
    default_style: str = "elegant_subtle_no_vulgarity"
    forbidden_endings: List[str] = Field(default_factory=lambda: ["bisous"]) # user-extendable via >>

    # Conversation register thresholds
    short_words_max: int = 80
    min_scene_punct: int = 6

    # Emoji control
    default_emoji_quota: int = 1


settings = Settings()

# ----------------------------
# In-memory stores (replace with DB later)
# ----------------------------

CONVERSATIONS: Dict[str, List[Message]] = {}  # session_id -> [Message]
SUMMARIES: Dict[str, str] = {}  # session_id -> rolling summary text
PREFERENCES: Dict[str, Dict[str, str]] = {}  # user_id -> {key:value}
INSTRUCTIONS: Dict[str, List[InstructionOverride]] = {}  # user_id -> overrides
MEMORIES: Dict[str, List[MemoryItem]] = {}  # user_id -> [MemoryItem]
SNAPSHOTS: Dict[str, Snapshot] = {}  # session_id -> Snapshot
MEMORY_USE_RECENCY: Dict[str, List[Tuple[str, float]]] = {}  # session_id -> [(mem_id, ts), ...]
BOOKS: Dict[str, Book] = {}
CHAPTERS: Dict[str, Chapter] = {}
CHAPTER_VERSIONS: Dict[str, List[ChapterVersion]] = {}
CHAPTER_EMB: Dict[str, List[float]] = {}  # embedding per chapter content (for later retrieval)

# Embeddings cache to reduce network calls
_EMB_CACHE: Dict[str, List[float]] = {}


# ----------------------------
# Utility functions
# ----------------------------


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
        "style_pref": book.style or settings.default_style,
    }


def render_chapter_system_prompt(book: Book, ctx: Dict[str, Any]) -> str:
    """System prompt specialized for chapter generation/editing; reuses the same rails."""
    rails = (
        "You are Henry (refined, mysterious). No vulgarity. No meta. French. "
        "Subtle emotion, lived details, realistic dialogue. Keep continuity with previous chapters."
    )
    rules = (
        "Write a novel chapter with a distinctive title. Mix narration and dialogues. "
        "End with a striking last line from Henry, no questions, no validation."
    )
    beats = ctx.get("outline_beat", "")
    neighbors = "; ".join([f"N{idx}:{beat}" for idx, beat in ctx.get("neighbor_beats", [])])
    themes = ", ".join(ctx.get("themes", []))
    style = ctx.get("style_pref", settings.default_style)

    lines = [
        rails,
        f"Book title: {book.title}",
        f"Themes: {themes}",
        f"Style: {style}",
        f"Current outline beat: {beats}",
        f"Neighbor beats: {neighbors}",
        "Use light spatio-temporal hints coherent with Europe/Paris timeframe.",
        rules,
    ]
    if ctx.get("long_facts"):
        lines.append(f"Author notes to weave subtly: {ctx['long_facts']}")
    if ctx.get("prev_chapters"):
        joined_prev = "\n---\n".join(ctx["prev_chapters"])
        lines.append(f"Continuity context (compressed):\n{joined_prev}")
    return "\n".join(lines)

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
        return t # handled elsewhere
    if t.lower().startswith("::author"):
        return t[len("::author"):].lstrip()
    if t.lower().startswith("::a"):
        return t[len("::a"):].lstrip()
    return text

CHAPTER_NEW_RE = _re.compile(r"^(?:chapitre|chapter)\s*(\d+)\s*:\s*(?:écris|write)\b", _re.I)
CHAPTER_EDIT_RE = _re.compile(r"^(?:chapitre|chapter)\s*(\d+)\s*:\s*(?:réécris|edit|rewrite)\b", _re.I)


def detect_chapter_intent(txt: str) -> Optional[Dict[str, Any]]:
    """Return {action, index}|None: lightweight convention for chat commands.
    Examples:
      - "Chapitre 3: écris une version plus sombre" -> {action:"generate", index:3, note:"une version plus sombre"}
      - "Chapitre 2: réécris en 1200 mots" -> {action:"edit", index:2, note:"en 1200 mots"}
    """
    t = txt.strip()
    m = CHAPTER_NEW_RE.match(t)
    if m:
        return {"action": "generate", "index": int(m.group(1)), "note": t[m.end():].strip()}
    m = CHAPTER_EDIT_RE.match(t)
    if m:
        return {"action": "edit", "index": int(m.group(1)), "note": t[m.end():].strip()}
    return None


# ----------------------------
# Conversation register detection
# ----------------------------

class ConvRegister(str):
    brevity = "brevity"   # short, poetic lines
    scene = "scene"       # narrative + dialogue


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

def embed(text: str) -> List[float]:
    """Call OpenAI embeddings with small in-memory cache."""
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    url = f"{OPENAI_BASE_URL}/embeddings"
    payload = {"model": OPENAI_EMB_MODEL, "input": text}
    try:
        r = requests.post(url, json=payload, headers=build_openai_headers(OPENAI_API_KEY), timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {r.text}")
        vec = r.json()["data"][0]["embedding"]
        _EMB_CACHE[text] = vec
        return vec
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings network error: {str(e)}")


def mmr_select(
        query_vec: np.ndarray,
        candidates: List[MemoryItem],
        k: int,
        lambda_mult: float,
        used_recent_ids: Optional[List[str]] = None
) -> List[MemoryItem]:
    """
    Maximal Marginal Relevance (simplified):
    - Promote items similar to the query (relevance)
    - Penalize items similar to already selected (diversity)
    - Light penalty for recently used ids to avoid repetition
    """
    if k <= 0 or not candidates:
        return []
    used_recent_ids = set(used_recent_ids or [])

    c_vecs = [np.array(m.embedding, dtype=float) for m in candidates]
    q_sims = [cosine_sim(query_vec, v) for v in c_vecs]

    selected: List[int] = []
    while len(selected) < min(k, len(candidates)):
        best_idx, best_score = None, -1e9
        for i, cand in enumerate(candidates):
            if i in selected:
                continue
            penalty = -0.05 if cand.id in used_recent_ids else 0.0
            if selected:
                max_sim_selected = max(cosine_sim(c_vecs[i], c_vecs[j]) for j in selected)
            else:
                max_sim_selected = 0.0
            score = lambda_mult * q_sims[i] - (1 - lambda_mult) * max_sim_selected + penalty
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is None:
            break
        selected.append(best_idx)

    return [candidates[i] for i in selected]


# ----------------------------
# Short-memory (rolling summary)
# ----------------------------

def get_or_refresh_summary(session_id: str) -> str:
    """
    Very small rolling summary that extracts motifs/constraints.
    Refreshed every N turns to reduce token bloat and repetition.
    """
    msgs = CONVERSATIONS.get(session_id, [])
    if not msgs:
        return ""
    if (len(msgs) % settings.summary_refresh_every_n_turns) != 0 and session_id in SUMMARIES:
        return SUMMARIES[session_id]

    # Look at the last few user/assistant messages
    last_user = [m.content for m in msgs if m.role == "user"][-5:]
    last_assist = [m.content for m in msgs if m.role == "assistant"][-5:]
    corpus = " ".join(last_user + last_assist).lower()

    motifs = []
    for k in ["lune", "lac", "pluie", "café", "été", "hiver", "orage", "neige"]:
        if k in corpus:
            motifs.append(k)

    snap = SNAPSHOTS.get(session_id)
    place = f"{snap.location}" if snap and snap.location else "—"
    summary = f"Tone:elegant/subtle; Place:{place}; Motifs:{', '.join(motifs) or '—'}."
    SUMMARIES[session_id] = summary
    return summary


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
    user_text: str,
    mode: str,
    snapshot_override: Optional[Snapshot],
) -> Tuple[ContextPackage, List[str], List[str]]:
    # imports & stores are assumed from original file
    prefs = PREFERENCES.get(user_id, {})
    active_instr = [i for i in INSTRUCTIONS.get(user_id, []) if i.active]
    instructions = {
        "style": settings.default_style,
        "forbidden_endings": settings.forbidden_endings,
        "bounds": ["no_vulgarity", "no_intrusive_secrets"],
        "preferences": prefs,
        "instruction_overrides": [{"key": i.rule_key, "value": i.rule_value} for i in active_instr],
    }

    snapshot = build_snapshot(session_id, snapshot_override)
    used_facets = [snapshot.selected_facet] if snapshot.selected_facet else []

    msgs = CONVERSATIONS.get(session_id, [])
    if mode == "author":
        k = settings.recent_messages_in_context_author
        top_k = settings.emb_top_k_author
    else:  # conversation
        k = settings.recent_messages_in_context_conversation
        top_k = settings.emb_top_k_conversation
    recent_msgs = [{"role": m.role, "content": m.content} for m in msgs[-k:]]

    short_memory = {
        "summary": get_or_refresh_summary(session_id),
        "recent_messages": recent_msgs,
    }

    long_list = MEMORIES.get(user_id, [])
    used_recent_ids = [mid for mid, _ in MEMORY_USE_RECENCY.get(session_id, [])[-settings.cooldown_memory_messages:]]

    # Encode query for retrieval
    q_vec = np.array(embed(user_text), dtype=float)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    selected_memories: List[MemoryItem] = []
    if top_k > 0 and long_list:
        selected_memories = mmr_select(
            query_vec=q_vec,
            candidates=long_list,
            k=top_k,
            lambda_mult=settings.mmr_lambda,
            used_recent_ids=used_recent_ids,
        )

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

def render_system_prompt_conversation(ctx: ContextPackage, register: str) -> str:
    snap = ctx.snapshot
    instr = ctx.instructions
    forb = ", ".join(instr.get("forbidden_endings", []))
    space_time_hint = f"{snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather} | facet:{snap.selected_facet}"

    if register == ConvRegister.brevity:
        reg_rules = (
            "Réponds en 1–4 phrases, brèves, élégantes, poétiques. "
            "Sans narration lourde; images et métaphores fines. 0–1 emoji maximum."
        )
    else:
        reg_rules = (
            "Poursuis en scène romanesque vivante mêlant descriptions (gestes, décors, atmosphère) et dialogues. "
            "Termine par une dernière phrase d’Henry, intime et marquante; jamais de validation ni question."
        )

    return "\n".join([
        "Tu es Henry : gentleman raffiné, mystérieux, attentif. Jamais de vulgarité.",
        f"Interdictions de fin: {forb}.",
        "Émojis très ponctuels (0–1).",
        f"Registre: {register}. {reg_rules}",
        f"Cohérence discrète avec lieu/saison/heure: {space_time_hint}.",
        "Aucune meta-intro ni meta-conclusion. N'exige jamais de secrets intrusifs.",
    ])


def render_system_prompt_author(ctx: ContextPackage) -> str:
    snap = ctx.snapshot
    instr = ctx.instructions
    forb = ", ".join(instr.get("forbidden_endings", []))
    space_time_hint = f"{snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather} | facet:{snap.selected_facet}"

    return "\n".join([
        "Tu es un écrivain expérimenté (hors personnage). Français élégant, crédible, maîtrisé.",
        "Objectif: écrire un chapitre romanesque inspiré du contexte de conversation et de la mémoire.",
        f"Interdictions de fin: {forb}.",
        "Mixe narration et dialogues, rythme vivant, pas de vulgarité.",
        "Conclure naturellement (pas de validation/question).",
        f"Cohérence discrète avec lieu/saison/heure: {space_time_hint}.",
        "Pas de meta. Titre sobre et distinctif en première ligne.",
    ])

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
# >> inline instruction handling
# ----------------------------

def apply_inline_instruction(user_id: str, text: str) -> None:
    cmd = text.strip()[2:].strip()
    if not cmd:
        return
    norm = cmd.lower()
    lst = INSTRUCTIONS.setdefault(user_id, [])

    # Example: >> Ne termine plus tes messages par "bisous".
    if "ne termine plus tes messages par" in norm and '"' in cmd:
        token = cmd.split('"')[1]
        if token:
            if token.lower() not in (s.lower() for s in settings.forbidden_endings):
                settings.forbidden_endings.append(token)
            lst.append(InstructionOverride(rule_key="forbidden_ending_add", rule_value=token))
        return

    # Example: >> Autorise 0 emoji.
    if "autorise 0 emoji" in norm or "zéro emoji" in norm or "zero emoji" in norm:
        lst.append(InstructionOverride(rule_key="emoji_quota", rule_value="0"))
        return

    # Store raw command
    lst.append(InstructionOverride(rule_key="raw", rule_value=cmd))


# ----------------------------
# Output post-processing (hard rails)
# ----------------------------

META_PATTERNS = [
    "merci pour ta confiance",
    "voici la réécriture",
    "veux-tu que je continue",
    "est-ce que ça te convient",
    "fin du chapitre",
    "do you want me to continue",
    "should i continue",
]

EMOJI_SET = set("😊😉😍🥰🤍✨💫🌙🔥🍷🌧️☕️")


def enforce_output_rules(text: str, ctx: ContextPackage, register: Optional[str] = None) -> str:
    out = (text or "").strip()
    # remove meta patterns (case-insensitive)
    for p in META_PATTERNS:
        out = re.sub(p, "", out, flags=re.IGNORECASE)

    # forbidden endings
    forb = [f.lower() for f in ctx.instructions.get("forbidden_endings", [])]
    low = out.lower().rstrip()
    for token in forb:
        if low.endswith(token):
            out = out[: -len(token)].rstrip(".!? \n")
            break

    # emoji quota
    quota = settings.default_emoji_quota
    for o in ctx.instructions.get("instruction_overrides", []):
        if o.get("key") == "emoji_quota":
            try:
                quota = int(o.get("value", str(quota)))
            except Exception:
                pass
    current_emojis = [c for c in out if c in EMOJI_SET]
    if len(current_emojis) > quota:
        # remove from end backwards
        to_remove = len(current_emojis) - quota
        i = len(out) - 1
        while i >= 0 and to_remove > 0:
            if out[i] in EMOJI_SET:
                out = out[:i] + out[i + 1 :]
                to_remove -= 1
            i -= 1

    # conversation scene must not end with question/validation
    if ctx.mode == "conversation" and register == ConvRegister.scene:
        if out.endswith("?"):
            out = out[:-1].rstrip() + "."
        tail_bad = [
            "veux-tu",
            "souhaites-tu",
            "je continue",
            "tu veux que je continue",
            "do you want me to continue",
            "should i continue",
        ]
        low = out.lower().rstrip()
        if any(low.endswith(t) for t in tail_bad):
            out = out.rstrip(".!?\n ") + "."

    return out.strip()


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Dear Gentle — Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# HTTP endpoints
# ----------------------------


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_mode = detect_mode(req.message, default_mode="conversation")
    raw_user_text = strip_mode_marker(req.message)

    # instruction mode: store preference/memory, append message, return empty output
    if user_mode == "instruction" and req.message.startswith(">>"):
        apply_inline_instruction(req.user_id, req.message)
        # also store in MEMORIES with a special tag for retrieval
        lst = MEMORIES.setdefault(req.user_id, [])
        vec = embed(req.message)
        lst.append(MemoryItem(
            id=str(uuid.uuid4()), user_id=req.user_id, text=req.message, embedding=vec, tags=["instruction"]
        ))
        CONVERSATIONS.setdefault(req.session_id, []).append(
            Message(role="user", content=req.message, mode=user_mode)
        )
        return ChatResponse(output="", mode="instruction", used_facets=[], used_memory_ids=[])

    # persist user message
    CONVERSATIONS.setdefault(req.session_id, []).append(
        Message(role="user", content=raw_user_text, mode=user_mode)
    )

    # build context
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        user_text=raw_user_text,
        mode=user_mode,
        snapshot_override=req.snapshot_override,
    )

    messages: List[Dict[str, str]] = []

    if user_mode == "author":
        sys = render_system_prompt_author(ctx)
        messages.append({"role": "system", "content": sys})
        # supply compact conversation digest for inspiration
        history = ctx.short_memory.get("recent_messages", [])
        digest_lines = [f"{m['role']}: {m['content']}" for m in history][-18:]
        user_payload = (
            "Écris un chapitre inspiré de cette conversation. Titre en première ligne.\n\n"
            + "\n".join(digest_lines)
            + f"\n\nConsignes de l'utilisateur:\n{raw_user_text}"
        )
        messages.append({"role": "user", "content": user_payload})
        raw_output = call_openai_chat(messages)
        output = enforce_output_rules(raw_output, ctx)

        # OPTIONAL: persist as chapter for the current session's book if exists
        book = BOOKS.get(req.session_id)
        if book:
            _persist_chapter_from_output(
                user_id=req.user_id,
                book_id=book.id,
                content=output,
                forced_index=None,
                used_facets=used_facets,
            )

    else:  # conversation
        register = detect_conv_register(raw_user_text)
        sys = render_system_prompt_conversation(ctx, register)
        messages.append({"role": "system", "content": sys})

        # Inject a few recent turns for continuity
        for m in ctx.short_memory.get("recent_messages", [])[-14:]:
            role = m["role"] if m["role"] in ("user", "assistant", "system") else "user"
            messages.append({"role": role, "content": m["content"]})

        messages.append({"role": "user", "content": raw_user_text})
        raw_output = call_openai_chat(messages)
        output = enforce_output_rules(raw_output, ctx, register=register)

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
    """Simple helper to store user preferences (as free-form key/values)."""
    d = PREFERENCES.setdefault(req.user_id, {})
    for it in req.items:
        d[it.key] = it.value
    return {"ok": True, "count": len(req.items)}


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


@app.get("/healthz")
def healthz():
    return {
        "ok": True
    }


@app.post("/api/book/upsert")
def upsert_book(book: Book):
    """Create or update a Book (id provided by client to keep things simple)."""
    if not book.id:
        raise HTTPException(status_code=400, detail="book.id is required")
    book.updated_at = time.time()
    BOOKS[book.id] = book
    return {"ok": True, "book_id": book.id}


@app.get("/api/chapters")
def list_chapters(book_id: str):
    items = [c for c in CHAPTERS.values() if c.book_id == book_id]
    items = sorted(items, key=lambda x: x.index)
    return {"ok": True, "items": items}


@app.get("/api/chapters/{chapter_id}")
def get_chapter(chapter_id: str):
    ch = CHAPTERS.get(chapter_id)
    if not ch:
        raise HTTPException(status_code=404, detail="chapter not found")
    return {"ok": True, "item": ch, "versions": CHAPTER_VERSIONS.get(chapter_id, [])}


@app.post("/api/chapters/generate")
def generate_chapter(req: ChapterGenRequest):
    book = BOOKS.get(req.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="book not found")

    # Build narrative context
    chap_ctx = build_chapter_context(req.user_id, book, req.chapter_index, req.use_prev_chapters)

    # Reuse snapshot/facets for coherence
    session_id = req.book_id  # lightweight: one session per book
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=session_id,
        user_text=req.prompt or f"Chapitre {req.chapter_index}",
        mode="chapter",
        snapshot_override=req.snapshot_override,
    )

    # Compose messages
    # pour les chapitres, on réutilise les rails "conversation" en mode scène
    sys = render_chapter_system_prompt(book, chap_ctx)
    sys = sys + "\n\n" + render_system_prompt_conversation(ctx, ConvRegister.scene)
    user_msg = (req.prompt or "Écris le chapitre demandé.")

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user_msg}]
    raw = call_openai_chat(messages)
    out = enforce_output_rules(raw, ctx)

    ch = Chapter(
        id=str(uuid.uuid4()),
        book_id=req.book_id,
        index=req.chapter_index,
        title=(out.splitlines()[0].strip() if out.strip() else f"Chapitre {req.chapter_index}"),
        content=out,
        model=OPENAI_CHAT_MODEL,
    )
    ch.summary = summarize_chapter(ch)
    CHAPTERS[ch.id] = ch

    # Create first version
    ver = ChapterVersion(id=str(uuid.uuid4()), chapter_id=ch.id, title=ch.title, content=ch.content, notes="v1")
    CHAPTER_VERSIONS.setdefault(ch.id, []).append(ver)

    # Embed for later retrieval
    try:
        CHAPTER_EMB[ch.id] = embed(ch.content)
    except HTTPException:
        CHAPTER_EMB[ch.id] = []

    # Mark facet usage for continuity
    snap = SNAPSHOTS.get(session_id) or build_snapshot(session_id, None)
    snap = mark_facets_used(snap, session_id, used_facets)
    SNAPSHOTS[session_id] = snap

    return {"ok": True, "chapter_id": ch.id, "used_facets": used_facets, "used_memory_ids": used_mem_ids, "item": ch}


@app.post("/api/chapters/{chapter_id}/edit")
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
    sys = render_chapter_system_prompt(book, chap_ctx)
    sys = sys + "\n\n" + render_system_prompt_conversation(ctx, ConvRegister.scene)

    edit_prompt = (
        "Réécris le chapitre en respectant les consignes d'édition suivantes (français) :\n"
        f"- {req.edit_instruction}\n"
        "- Conserve la continuité, ne change pas les événements clés sauf si demandé.\n"
        "- Garde une dernière ligne percutante par Henry; pas de question ni validation.\n"
        "- Titre conservé ou amélioré, mais sobre.\n\n"
        f"TEXTE ACTUEL:\n{compress_text_for_context(ch.content, 3200)}\n"
    )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": edit_prompt}]
    raw = call_openai_chat(messages)
    out = enforce_output_rules(raw, ctx)

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


# ----------------------------
# (6) Optional: route chat commands to chapter endpoints (backward compatible UX)
# ----------------------------

@app.post("/api/chat+chapters", response_model=ChatResponse)
def chat_with_chapters(req: ChatRequest):
    """Wrapper endpoint: if the user asks about chapters explicitly, route to generation/editing; else fallback to /api/chat logic."""
    intent = detect_chapter_intent(req.message)
    if intent:
        # Map book/session 1:1 for simplicity; client should pass session_id as book_id here.
        book = BOOKS.get(req.session_id)
        if not book:
            raise HTTPException(status_code=404, detail="book (session_id) not found; call /api/book/upsert first")
        if intent["action"] == "generate":
            gen = ChapterGenRequest(
                user_id=req.user_id,
                book_id=book.id,
                chapter_index=intent["index"],
                prompt=intent.get("note") or None,
                use_prev_chapters=2,
                snapshot_override=req.snapshot_override,
            )
            result = generate_chapter(gen)  # type: ignore
            return ChatResponse(output=result["item"].content, mode="chapter", used_facets=result["used_facets"], used_memory_ids=result["used_memory_ids"])
        else:
            # find chapter by index in this book
            candidates = [c for c in CHAPTERS.values() if c.book_id == book.id and c.index == intent["index"]]
            if not candidates:
                raise HTTPException(status_code=404, detail="chapter not found for edit")
            ch = sorted(candidates, key=lambda x: (x.updated_at or 0))[-1]
            result = edit_chapter(ch.id, ChapterEditRequest(user_id=req.user_id, chapter_id=ch.id, edit_instruction=intent.get("note") or "Améliore le rythme."))  # type: ignore
            return ChatResponse(output=result["item"].content, mode="rewrite", used_facets=result["used_facets"], used_memory_ids=result["used_memory_ids"])

    # Fallback: regular chat
    return chat(req)

# ----------------------------
# Notes for the front-end (FYI)
# ----------------------------
# - If /api/chat returns mode == "instruction" and output == "",
#   just ignore rendering and keep the local UX: the instruction has been applied.
# - You can expose a toggle to show "used_facets" and "used_memory_ids" for debugging.
# - When using ++, the rewrite scaffold ensures consistency (chapter title + closure).
# - To reduce repetition further, consider seeding meaningful memories:
#     POST /api/seed/memories with recurring motifs (places, tastes, references).
# - CORS: set FRONT_ORIGIN env var to your deployed front URL.

# ----------------------------
# (7) Minimal OpenAPI examples (to paste into README or to test quickly)
# ----------------------------
"""
POST /api/book/upsert
{
    "id": "book-123",
    "user_id": "u1",
    "title": "Romance secrète",
    "outline": [
    "Rencontre au bord du lac",
    "Le message retrouvé",
    "Premier aveu",
    "La distance",
    "Retrouvailles"
    ],
    "themes": ["lac d'Annecy", "été", "mystère", "tendresse"],
    "style": "elegant_subtle_no_vulgarity"
}


POST /api/chapters/generate
{
    "user_id": "u1",
    "book_id": "book-123",
    "chapter_index": 1,
    "prompt": "Chapitre d'ouverture, 1200-1500 mots, rythme posé.",
    "use_prev_chapters": 0
}


POST /api/chapters/{chapter_id}/edit
{
    "user_id": "u1",
    "chapter_id": "<id renvoyé>",
    "edit_instruction": "Réduis de 15%, rend plus allusif, garde la dernière image."
}


GET /api/chapters?book_id=book-123
GET /api/chapters/{chapter_id}
"""