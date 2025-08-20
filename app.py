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
import time
import uuid
import math
import datetime as dt
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Environment & settings
# ----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
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
    recent_messages_in_context_short: int = 20
    recent_messages_in_context_chapter: int = 40
    recent_messages_in_context_rewrite: int = 20

    # Vector search top-k (long memory)
    emb_top_k_short: int = 2
    emb_top_k_chapter: int = 5
    emb_top_k_rewrite: int = 5

    # Facet scheduling
    facet_cooldown_messages: int = 3
    max_facets_per_output: int = 1

    # Memory reuse cooldown (by last N uses in session)
    cooldown_memory_messages: int = 3

    # MMR control
    mmr_lambda: float = 0.5  # relevance vs. diversity

    # How often to refresh rolling summary
    summary_refresh_every_n_turns: int = 5

    # Persona/style rails
    default_style: str = "elegant_subtle_no_vulgarity"
    forbidden_endings: List[str] = Field(default_factory=lambda: ["bisous"])
    allowed_dares: List[str] = Field(default_factory=lambda: ["clothing", "sentence", "scent", "book_excerpt"])

settings = Settings()

# ----------------------------
# Data models
# ----------------------------

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    ts: float = Field(default_factory=lambda: time.time())
    mode: Optional[str] = None

class Snapshot(BaseModel):
    # Minimal "world state" to keep Henry coherent (place/season/time/weather)
    location: Optional[Dict[str, str]] = None  # {"city":"Annecy","country":"France"}
    datetime_local_iso: Optional[str] = None
    season: Optional[str] = None               # "summer", "winter", ...
    time_of_day: Optional[str] = None          # "morning","afternoon","evening","night"
    weather: Optional[Dict[str, str]] = None   # {"condition":"mild_evening","temperature_c":27}
    contextual_facets: List[str] = []
    last_mentioned_facets: List[Dict[str, str]] = []  # [{"facet":"lac d’Annecy","ts":"..."}]
    cultural_refs: List[str] = []              # user-provided cultural references
    selected_facet: Optional[str] = None       # chosen once per turn

class Preference(BaseModel):
    key: str
    value: str

class InstructionOverride(BaseModel):
    # Persistent knobs controlled by >> commands (per user)
    rule_key: str
    rule_value: str
    active: bool = True

class MemoryItem(BaseModel):
    # Long-term memory chunk
    id: str
    user_id: str
    text: str
    embedding: List[float]
    tags: List[str] = []
    source: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())

class ContextPackage(BaseModel):
    mode: str
    instructions: Dict[str, object]
    snapshot: Snapshot
    short_memory: Dict[str, object]
    long_memory: Dict[str, object]  # {"facts":[{"id":..., "text":...}]}

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    snapshot_override: Optional[Snapshot] = None

class ChatResponse(BaseModel):
    output: str
    mode: str
    used_facets: List[str] = []
    used_memory_ids: List[str] = []

class SeedPrefRequest(BaseModel):
    user_id: str
    items: List[Preference]

class SeedMemoryRequest(BaseModel):
    user_id: str
    texts: List[str]

class SetSnapshotRequest(BaseModel):
    session_id: str
    snapshot: Snapshot

# ----------------------------
# In-memory stores (replace with DB later)
# ----------------------------

CONVERSATIONS: Dict[str, List[Message]] = {}   # session_id -> [Message]
SUMMARIES: Dict[str, str] = {}                 # session_id -> rolling summary text
PREFERENCES: Dict[str, Dict[str, str]] = {}    # user_id -> {key:value}
INSTRUCTIONS: Dict[str, List[InstructionOverride]] = {}  # user_id -> overrides
MEMORIES: Dict[str, List[MemoryItem]] = {}     # user_id -> [MemoryItem]
SNAPSHOTS: Dict[str, Snapshot] = {}            # session_id -> Snapshot
MEMORY_USE_RECENCY: Dict[str, List[Tuple[str, float]]] = {}  # session_id -> [(mem_id, ts), ...]

# Embeddings cache to reduce network calls
_EMB_CACHE: Dict[str, List[float]] = {}

# ----------------------------
# Utility functions
# ----------------------------

def paris_now() -> dt.datetime:
    """Current time in Europe/Paris."""
    return dt.datetime.now(ZoneInfo("Europe/Paris"))

def now_iso_paris() -> str:
    """ISO datetime in Europe/Paris."""
    return paris_now().isoformat()

def infer_season_from_date(d: dt.datetime) -> str:
    """Simple season inference for EU (meteorological seasons)."""
    if d.month in (12, 1, 2):
        return "winter"
    if d.month in (3, 4, 5):
        return "spring"
    if d.month in (6, 7, 8):
        return "summer"
    return "autumn"

def infer_time_of_day(d: dt.datetime) -> str:
    """Coarse-grained moment of the day."""
    h = d.hour
    if 5 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "evening"
    return "night"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-safety."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def build_openai_headers() -> Dict[str, str]:
    """HTTP headers for OpenAI API."""
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

# ----------------------------
# Mode detection
# ----------------------------

def detect_mode(user_text: str, default_mode: str = "short") -> str:
    """Decide mode based on leading tokens and narrative density."""
    t = user_text.strip()
    if t.startswith("++"):
        return "rewrite"
    if t.startswith(">>"):
        return "instruction"
    # Heuristic for "chapter": long, punctuated, and not a question
    words = len(t.split())
    punct = sum(t.count(p) for p in [".", ",", ";", "—", "…", "!"])
    if words > 120 and punct >= 6 and not t.endswith("?"):
        return "chapter"
    return default_mode

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
        r = requests.post(url, json=payload, headers=build_openai_headers(), timeout=30)
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
    snapshot_override: Optional[Snapshot]
) -> Tuple[ContextPackage, List[str], List[str]]:
    """
    Assemble:
    - Instructions (persona rails + user overrides)
    - Snapshot (with a single selected facet)
    - Short memory (summary + recent messages)
    - Long memory (embeddings search + MMR + reuse cooldown)
    """
    prefs = PREFERENCES.get(user_id, {})
    active_instr = [i for i in INSTRUCTIONS.get(user_id, []) if i.active]
    instructions = {
        "style": settings.default_style,
        "forbidden_endings": settings.forbidden_endings,
        "bounds": ["no_vulgarity", "no_intrusive_secrets"],
        "allowed_dares": settings.allowed_dares,
        "preferences": prefs,
        "instruction_overrides": [{"key": i.rule_key, "value": i.rule_value} for i in active_instr],
    }

    snapshot = build_snapshot(session_id, snapshot_override)
    used_facets = [snapshot.selected_facet] if snapshot.selected_facet else []

    msgs = CONVERSATIONS.get(session_id, [])
    if mode == "short":
        k = settings.recent_messages_in_context_short
    elif mode == "chapter":
        k = settings.recent_messages_in_context_chapter
    else:
        k = settings.recent_messages_in_context_rewrite
    recent_msgs = [{"role": m.role, "content": m.content} for m in msgs[-k:]]

    short_memory = {
        "summary": get_or_refresh_summary(session_id),
        "recent_messages": recent_msgs,
    }

    long_list = MEMORIES.get(user_id, [])
    used_recent_ids = [mid for mid, _ in MEMORY_USE_RECENCY.get(session_id, [])[-settings.cooldown_memory_messages:]]
    # Encode query with real embeddings
    q_vec = np.array(embed(user_text), dtype=float)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    if mode == "short":
        top_k = settings.emb_top_k_short
    elif mode == "chapter":
        top_k = settings.emb_top_k_chapter
    else:
        top_k = settings.emb_top_k_rewrite

    selected_memories: List[MemoryItem] = []
    if top_k > 0 and long_list:
        selected_memories = mmr_select(
            query_vec=q_vec,
            candidates=long_list,
            k=top_k,
            lambda_mult=settings.mmr_lambda,
            used_recent_ids=used_recent_ids
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

def render_system_prompt(ctx: ContextPackage) -> str:
    """
    Tight, mode-aware system prompt enforcing Henry's rails:
    - No vulgarity, no meta
    - Sparse emojis
    - Mode-specific output constraints
    - Light use of space-time hints and cultural references
    """
    instr = ctx.instructions
    snap = ctx.snapshot
    facts = ctx.long_memory.get("facts", [])
    bounds = ", ".join(instr.get("bounds", []))
    forbidden = ", ".join(instr.get("forbidden_endings", []))
    prefs = instr.get("preferences", {})
    cref = ", ".join(snap.cultural_refs or [])

    mode_rules = {
        "short": (
            "Write brief, elegant, poetic lines (1–4 sentences). "
            "No heavy narration. Subtle metaphors. Never meta. 0–1 emoji maximum."
        ),
        "chapter": (
            "Write a lived novel-like scene mixing narration and dialogue. "
            "Rich but controlled imagery; credible and realistic. "
            "End with a last intimate, striking line by Henry. "
            "Never end with a question or a validation."
        ),
        "rewrite": (
            "Rewrite as a literary chapter with a title. "
            "Blend narration and dialogues. "
            "Close naturally without questions or validations."
        ),
    }

    space_time_hint = f"{snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather} | facet:{snap.selected_facet}"

    lines = [
        "You are Henry: refined, mysterious, attentive; elegant and subtle; no vulgarity.",
        f"Global bounds: {bounds}. Forbidden endings: {forbidden}.",
        "Emojis are optional and very sparse; never more than one.",
        f"Current mode: {ctx.mode}. Mode rules: {mode_rules.get(ctx.mode, mode_rules['short'])}",
        f"Coherence: discreetly adapt to place/season/time: {space_time_hint}. Use at most one soft reference.",
        "Never use meta-introductions or meta-conclusions. Never ask for intrusive secrets.",
        "Dares are poetic and playful only (clothing, sentence, scent, book_excerpt).",
    ]
    if prefs:
        lines.append(f"User preferences (weave lightly if relevant): {prefs}")
    if cref:
        lines.append(f"Cultural references (use subtly if helpful): {cref}")
    if facts:
        lines.append("Contextual facts (optional, weave naturally): " + " | ".join([f["text"] for f in facts]))

    return "\n".join(lines)

# ----------------------------
# OpenAI call (robust)
# ----------------------------

def call_openai_chat(messages: List[Dict[str, str]], retries: int = 2) -> str:
    """Robust chat call with exponential backoff."""
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.4,
    }
    headers = build_openai_headers()

    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            last_err = r.text
        except requests.RequestException as e:
            last_err = str(e)
        # Backoff: 0.8s, 1.6s, 3.2s...
        time.sleep(0.8 * (2 ** i))

    raise HTTPException(status_code=502, detail=f"OpenAI chat error: {last_err}")

# ----------------------------
# ++ rewrite payload scaffold
# ----------------------------

def build_rewrite_payload(req: ChatRequest, ctx: ContextPackage) -> str:
    """
    Provide a clear scaffold for the literary rewrite:
    - Chapter title
    - Narration + dialogues
    - Natural closure (no validation/question at the end)
    """
    history = ctx.short_memory.get("recent_messages", [])
    lines = [f"{m['role']}: {m['content']}" for m in history][-14:]
    digest = "\n".join(lines)

    return (
        "++ LITERARY REWRITE REQUEST\n"
        "Output requirements:\n"
        "- Add a chapter title (e.g., 'Chapitre I — ...').\n"
        "- Blend narration and dialogues.\n"
        "- Close naturally without questions or validations.\n\n"
        f"Conversation digest:\n{digest}\n\n"
        f"User says:\n{req.message}"
    )

# ----------------------------
# >> inline instruction handling
# ----------------------------

def apply_inline_instruction(user_id: str, text: str) -> None:
    """
    Parse basic >> instructions and persist them as InstructionOverride.
    We keep this simple and extend with patterns as needed.
    """
    cmd = text.strip()[2:].strip()
    if not cmd:
        return
    norm = cmd.lower()
    lst = INSTRUCTIONS.setdefault(user_id, [])

    # Example: >> Ne termine plus tes messages par "bisous".
    if "ne termine plus tes messages par" in norm and '"' in cmd:
        token = cmd.split('"')[1]
        if token:
            # Add to settings (runtime process) and record an override
            if token.lower() not in (s.lower() for s in settings.forbidden_endings):
                settings.forbidden_endings.append(token)
            lst.append(InstructionOverride(rule_key="forbidden_ending_add", rule_value=token))
        return

    # Example: >> Autorise 0 emoji.
    if "autorise 0 emoji" in norm or "zéro emoji" in norm or "zero emoji" in norm:
        lst.append(InstructionOverride(rule_key="emoji_quota", rule_value="0"))
        return

    # Fallback: store raw command for future use
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

SEASON_INCOHERENCE = {
    # very light, non-blocking replacements; keep this small and safe
    "summer": ["cheminée", "feu de cheminée", "neige abondante"],
    "winter": ["canicule", "orages d’été", "terrasse en short"],
}

def enforce_output_rules(text: str, ctx: ContextPackage) -> str:
    """
    Enforce:
    - No forbidden endings (e.g., 'bisous')
    - No meta-intros/outros
    - Emoji quota (default 1; can be overridden)
    - Chapter/rewrite: no question/validation ending
    - Tiny seasonal coherence guardrail (soft replace)
    """
    out = (text or "").strip()
    low = out.lower()

    # 1) Remove meta phrases (case-insensitive, simple replace)
    for p in META_PATTERNS:
        low = low.replace(p, "")
    # Rebuild 'out' from 'low' with original casing strategy:
    # To keep it simple, if metas were present we just strip them by indices.
    # Here we simply take 'low' to ensure removal; cost: lowercased output.
    # Better approach: use regex with re.IGNORECASE and apply on original.
    # We'll implement a cleaner pass below using the original string.
    import re
    for p in META_PATTERNS:
        out = re.sub(p, "", out, flags=re.IGNORECASE)

    # 2) Trim forbidden ending tokens (exact suffix check, case-insensitive)
    forb = [f.lower() for f in ctx.instructions.get("forbidden_endings", [])]
    for token in forb:
        if out.lower().endswith(token):
            out = out[: -len(token)].rstrip(".! \n")

    # 3) Emoji quota
    quota = 1
    for o in ctx.instructions.get("instruction_overrides", []):
        if o.get("key") == "emoji_quota":
            try:
                quota = int(o.get("value", "1"))
            except:
                quota = 1
    # Naive emoji set (controlled and small on purpose)
    EMOJI_SET = set("😊😉😍🥰🤍✨💫🌙🔥🍷🌧️☕️")
    # Remove surplus emojis from the end backwards
    current_emojis = [c for c in out if c in EMOJI_SET]
    if len(current_emojis) > quota:
        to_remove = len(current_emojis) - quota
        i = len(out) - 1
        while i >= 0 and to_remove > 0:
            if out[i] in EMOJI_SET:
                out = out[:i] + out[i+1:]
                to_remove -= 1
            i -= 1

    # 4) Chapter/Rewrite ending constraints
    if ctx.mode in ("chapter", "rewrite"):
        if out.endswith("?"):
            out = out[:-1].rstrip() + "."
        forbidden_tails = [
            "veux-tu", "souhaites-tu", "je continue", "tu veux que je continue",
            "do you want me to continue", "should i continue"
        ]
        tail = out.lower().rstrip()
        if any(tail.endswith(t) for t in forbidden_tails):
            out = out.rstrip(".!?\n ") + "."

    # 5) Seasonal soft replacement (very conservative)
    season = (ctx.snapshot.season or "").lower()
    inco = SEASON_INCOHERENCE.get(season, [])
    for word in inco:
        out = re.sub(word, "brise légère", out, flags=re.IGNORECASE)

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
    """
    Main chat endpoint.
    Flow:
      1) Detect mode (short/chapter/rewrite/instruction)
      2) If >> instruction: apply and return empty output (UI can ignore)
      3) Append user message
      4) Build context (snapshot, memories, summary, instructions)
      5) Build system + user payload (with ++ scaffold if needed)
      6) Call LLM
      7) Post-process output (rails)
      8) Persist assistant message, mark facets & memory recency
    """
    # 1) Mode detection
    user_mode = detect_mode(req.message, default_mode="short")

    # 2) Process >> instruction immediately (no LLM call)
    if user_mode == "instruction" and req.message.startswith(">>"):
        apply_inline_instruction(req.user_id, req.message)
        # Important: still append the user message for history,
        # so the summary remains faithful to what happened.
        CONVERSATIONS.setdefault(req.session_id, []).append(
            Message(role="user", content=req.message, mode=user_mode)
        )
        # Return an empty assistant message; front can ignore
        return ChatResponse(output="", mode="instruction", used_facets=[], used_memory_ids=[])

    # 3) Append user message
    CONVERSATIONS.setdefault(req.session_id, []).append(
        Message(role="user", content=req.message, mode=user_mode)
    )

    # 4) Build context
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        user_text=req.message,
        mode=user_mode if user_mode != "instruction" else "short",
        snapshot_override=req.snapshot_override
    )

    # 5) Build messages for the LLM
    system_prompt = render_system_prompt(ctx)
    user_payload = req.message
    if user_mode == "rewrite":
        user_payload = build_rewrite_payload(req, ctx)

    messages = [{"role": "system", "content": system_prompt}]

    # Inject recent turns in order so the model truly "remembers"
    for m in ctx.short_memory.get("recent_messages", []):
        role = m["role"]
        # Safety: only allow roles known by the API
        role = role if role in ("user", "assistant", "system") else "user"
        messages.append({"role": role, "content": m["content"]})

    # Finally the new user message (this turn)
    messages.append({"role": "user", "content": user_payload})
    # 6) Call LLM
    raw_output = call_openai_chat(messages)

    # 7) Post-process output to enforce rails
    output = enforce_output_rules(raw_output, ctx)

    # 8) Persist assistant message
    CONVERSATIONS[req.session_id].append(
        Message(role="assistant", content=output, mode=user_mode)
    )

    # Mark facets
    snap = SNAPSHOTS.get(req.session_id) or build_snapshot(req.session_id, None)
    snap = mark_facets_used(snap, req.session_id, used_facets)
    SNAPSHOTS[req.session_id] = snap

    # Mark memory recency
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