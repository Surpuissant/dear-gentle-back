# app.py
# FastAPI MVP for "Dear Gentle" orchestration.
# Notes:
# - Names/comments in English as requested.
# - In-memory stores for demo; replace with Supabase repositories when ready.

import os
import math
import time
import uuid
import datetime as dt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tiktoken  # optional, used for token budgeting if you want
import requests

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Settings (tune to taste)
# ----------------------------

class Settings(BaseModel):
    # mode policies
    recent_messages_in_context_short: int = 8   # you asked for more context → raise to 8
    recent_messages_in_context_chapter: int = 6
    recent_messages_in_context_rewrite: int = 6

    emb_top_k_short: int = 2       # conditional (we can skip if none found)
    emb_top_k_chapter: int = 5
    emb_top_k_rewrite: int = 5

    facet_cooldown_messages: int = 3
    max_facets_per_output: int = 1
    cooldown_memory_messages: int = 3

    mmr_lambda: float = 0.5  # balance between relevance and diversity

    summary_refresh_every_n_turns: int = 5

    # OpenAI
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"  # pick a fast, good-enough model
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Style & rules
    default_style: str = "elegant_subtle_no_vulgarity"
    forbidden_endings: List[str] = Field(default_factory=lambda: ["bisous"])
    allowed_dares: List[str] = Field(default_factory=lambda: ["clothing", "sentence", "scent", "book_excerpt"])

settings = Settings()

# ----------------------------
# Data models (Pydantic)
# ----------------------------

class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    ts: float = Field(default_factory=lambda: time.time())
    mode: Optional[str] = None

class Snapshot(BaseModel):
    location: Optional[Dict[str, str]] = None  # {"city":"Annecy","country":"France"}
    datetime_local_iso: Optional[str] = None
    season: Optional[str] = None               # "summer", "winter", ...
    time_of_day: Optional[str] = None          # "morning","evening",...
    weather: Optional[Dict[str, str]] = None   # {"condition":"warm_thunderstorms","temperature_c":27}
    contextual_facets: List[str] = []
    last_mentioned_facets: List[Dict[str, str]] = []  # [{"facet":"lac d’Annecy","ts":"..."}]

class Preference(BaseModel):
    key: str
    value: str

class InstructionOverride(BaseModel):
    rule_key: str
    rule_value: str
    active: bool = True

class MemoryItem(BaseModel):
    id: str
    user_id: str
    text: str
    embedding: List[float]
    tags: List[str] = []
    source: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    cooldown_state: Dict[str, Optional[str]] = Field(default_factory=lambda: {"last_used_ts": None, "uses_recent": 0})

class ContextPackage(BaseModel):
    mode: str
    instructions: Dict[str, object]
    snapshot: Snapshot
    short_memory: Dict[str, object]
    long_memory: Dict[str, object]

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

# ----------------------------
# In-memory stores (replace with Supabase later)
# ----------------------------

CONVERSATIONS: Dict[str, List[Message]] = {}   # key = session_id
SUMMARIES: Dict[str, str] = {}                 # session_id -> summary text
PREFERENCES: Dict[str, Dict[str, str]] = {}    # user_id -> {key:value}
INSTRUCTIONS: Dict[str, List[InstructionOverride]] = {}  # user_id -> overrides
MEMORIES: Dict[str, List[MemoryItem]] = {}     # user_id -> list of memory items
SNAPSHOTS: Dict[str, Snapshot] = {}            # session_id -> snapshot
MEMORY_USE_RECENCY: Dict[str, List[Tuple[str, float]]] = {}  # session_id -> [(mem_id, ts), ...]

# ----------------------------
# Utilities
# ----------------------------

def infer_season_from_date(d: dt.datetime) -> str:
    m = d.month
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5): return "spring"
    if m in (6, 7, 8): return "summer"
    return "autumn"

def infer_time_of_day(d: dt.datetime) -> str:
    h = d.hour
    if 5 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "evening"
    return "night"

def now_iso_paris() -> str:
    # naive implementation; you can switch to pytz/zoneinfo
    return dt.datetime.now().isoformat()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0: return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

def build_openai_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

# ----------------------------
# Modes router
# ----------------------------

def detect_mode(user_text: str, default_mode: str = "short") -> str:
    txt = user_text.strip()
    if txt.startswith("++"):
        return "rewrite"
    if txt.startswith(">>"):
        return "instruction"
    # heuristic: long narrative → chapter
    if len(txt) > 280 and any(p in txt for p in [". ", ", ", "; ", "\n"]) and not txt.endswith("?"):
        return "chapter"
    return default_mode

# ----------------------------
# Mention scheduler (facets)
# ----------------------------

def schedule_facets(snapshot: Snapshot, session_id: str) -> List[str]:
    """Return up to max_facets_per_output facets that are not in cooldown."""
    facets = []
    # Priority order: time_of_day, season, weather.condition, contextual_facets
    candidates = []

    if snapshot.time_of_day:
        candidates.append(("time_of_day", snapshot.time_of_day))
    if snapshot.season:
        candidates.append(("season", snapshot.season))
    if snapshot.weather and snapshot.weather.get("condition"):
        candidates.append(("weather", snapshot.weather["condition"]))
    for cf in snapshot.contextual_facets:
        candidates.append(("contextual", cf))

    # cooldown check using last_mentioned_facets
    last_map = {x["facet"]: x["ts"] for x in (snapshot.last_mentioned_facets or [])}
    allowed = []
    for kind, facet in candidates:
        last_ts = last_map.get(facet)
        recent_msgs = len(CONVERSATIONS.get(session_id, []))
        # simple proxy: use message count for cooldown gating
        # we mark ts as message index via len() string for MVP
        if last_ts is None:
            allowed.append(facet)
        else:
            try:
                last_idx = int(last_ts)  # if stored as index
            except:
                last_idx = 0
            if (recent_msgs - last_idx) >= settings.facet_cooldown_messages:
                allowed.append(facet)

    # Pick at most max_facets_per_output
    return allowed[:settings.max_facets_per_output]

def mark_facets_used(snapshot: Snapshot, session_id: str, used: List[str]) -> Snapshot:
    """Record facet usage as a message-count index (cheap and cheerful)."""
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
# Memory selection (MMR)
# ----------------------------

def encode_text_to_vec(text: str) -> np.ndarray:
    # For MVP: fake encoder with hashing → replace with real embeddings (OpenAI text-embedding-3-small)
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.normal(size=(256,))
    return vec / np.linalg.norm(vec)

def mmr_select(
    query_vec: np.ndarray,
    candidates: List[MemoryItem],
    k: int,
    lambda_mult: float = 0.5,
    used_recent_ids: Optional[List[str]] = None
) -> List[MemoryItem]:
    """Maximal Marginal Relevance selection to ensure diversity."""
    if k <= 0 or not candidates:
        return []

    used_recent_ids = set(used_recent_ids or [])
    # pre-compute sim(query, doc)
    c_vecs = [np.array(m.embedding, dtype=float) for m in candidates]
    q_sims = [cosine_sim(query_vec, v) for v in c_vecs]

    selected: List[int] = []
    while len(selected) < min(k, len(candidates)):
        best_idx = None
        best_score = -1e9
        for i, cand in enumerate(candidates):
            if i in selected:
                continue
            # penalize recent reuse
            penalty = -0.05 if cand.id in used_recent_ids else 0.0
            # diversity term: max sim to already selected
            if selected:
                max_sim_selected = max(
                    cosine_sim(c_vecs[i], c_vecs[j]) for j in selected
                )
            else:
                max_sim_selected = 0.0
            score = lambda_mult * q_sims[i] - (1 - lambda_mult) * max_sim_selected + penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)

    return [candidates[i] for i in selected]

# ----------------------------
# Short-memory management (summary)
# ----------------------------

def get_or_refresh_summary(session_id: str) -> str:
    msgs = CONVERSATIONS.get(session_id, [])
    if not msgs:
        return ""
    # refresh policy
    if (len(msgs) % settings.summary_refresh_every_n_turns) != 0 and session_id in SUMMARIES:
        return SUMMARIES[session_id]

    # Very small heuristic summarizer (placeholder for an LLM call if you want)
    # Extract key motifs and constraints.
    user_texts = [m.content for m in msgs if m.role == "user"][-5:]
    assistant_texts = [m.content for m in msgs if m.role == "assistant"][-5:]
    motifs = []
    if any("lune" in t.lower() for t in user_texts + assistant_texts):
        motifs.append("moon motif")
    if any("lac" in t.lower() for t in user_texts + assistant_texts):
        motifs.append("lake motif")
    summary = f"Tone: elegant, subtle. Motifs: {', '.join(motifs) or '—'}."
    SUMMARIES[session_id] = summary
    return summary

# ----------------------------
# Context builder
# ----------------------------

def build_snapshot(session_id: str, override: Optional[Snapshot]) -> Snapshot:
    snap = SNAPSHOTS.get(session_id)
    if not snap:
        # default snapshot
        d = dt.datetime.now()
        snap = Snapshot(
            location={"city": "Annecy", "country": "France"},
            datetime_local_iso=d.isoformat(),
            season=infer_season_from_date(d),
            time_of_day=infer_time_of_day(d),
            weather={"condition": "mild_evening"},
            contextual_facets=["terrasse au bord du lac"]
        )
    if override:
        # shallow merge for MVP
        data = snap.dict()
        for k, v in override.dict(exclude_none=True).items():
            data[k] = v
        snap = Snapshot(**data)

    # choose facets for this output (respect cooldown)
    chosen = schedule_facets(snap, session_id)
    # write chosen into snapshot so prompt can reference it explicitly
    if chosen:
        # we just tag the first; you can keep list if you want
        snap_dict = snap.dict()
        snap_dict["selected_facet_for_this_output"] = chosen[0]
        snap = Snapshot(**{k: v for k, v in snap_dict.items() if k in Snapshot.__fields__.keys()})
    return snap

def build_context(
    user_id: str,
    session_id: str,
    user_text: str,
    mode: str,
    snapshot_override: Optional[Snapshot]
) -> Tuple[ContextPackage, List[str], List[str]]:
    # A) Instructions / preferences
    prefs = PREFERENCES.get(user_id, {})
    active_instr = [i for i in INSTRUCTIONS.get(user_id, []) if i.active]
    instructions = {
        "style": settings.default_style,
        "forbidden_endings": settings.forbidden_endings,
        "bounds": ["no_vulgarity", "no_intrusive_secrets"],
        "allowed_dares": settings.allowed_dares,
        "preferences": prefs,
        "instruction_overrides": [{ "key": i.rule_key, "value": i.rule_value } for i in active_instr]
    }

    # B) Snapshot
    snapshot = build_snapshot(session_id, snapshot_override)
    used_facets = schedule_facets(snapshot, session_id)

    # C) Short memory
    msgs = CONVERSATIONS.get(session_id, [])
    if mode == "short":
        k = settings.recent_messages_in_context_short
    elif mode == "chapter":
        k = settings.recent_messages_in_context_chapter
    else:
        k = settings.recent_messages_in_context_rewrite
    recent_msgs = [m.dict() for m in msgs[-k:]]

    short_memory = {
        "summary": get_or_refresh_summary(session_id),
        "recent_messages": [{"role": m["role"], "content": m["content"]} for m in recent_msgs]
    }

    # D) Long memory via embeddings (MMR)
    long_list = MEMORIES.get(user_id, [])
    used_recent_ids = [mid for mid, _ in MEMORY_USE_RECENCY.get(session_id, [])[-settings.cooldown_memory_messages:]]
    query_vec = encode_text_to_vec(user_text)

    if mode == "short":
        top_k = settings.emb_top_k_short
    elif mode == "chapter":
        top_k = settings.emb_top_k_chapter
    else:
        top_k = settings.emb_top_k_rewrite

    selected_memories: List[MemoryItem] = []
    if top_k > 0 and long_list:
        selected_memories = mmr_select(
            query_vec=query_vec,
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
        long_memory={"facts": [{"id": m.id, "text": m.text} for m in selected_memories]}
    )
    used_mem_ids = [m.id for m in selected_memories]
    return ctx, used_facets, used_mem_ids

# ----------------------------
# LLM call (OpenAI, simple non-streaming for MVP)
# ----------------------------

def render_system_prompt(ctx: ContextPackage) -> str:
    """Compact, deterministic system prompt for Henry."""
    instr = ctx.instructions
    snap = ctx.snapshot
    facts = ctx.long_memory.get("facts", [])
    sel_facet = None
    # we stored "selected_facet_for_this_output" transiently during scheduling; pass as hint
    # We'll read from snapshot via getattr trick if needed; for MVP we just mention time_of_day/season/weather/context facet in text prompt.
    lines = [
        "You are Henry: refined, mysterious, attentive; elegant and subtle; no vulgarity.",
        f"Style: {instr.get('style')}. Bounds: {', '.join(instr.get('bounds', []))}.",
        f"Forbidden endings: {', '.join(instr.get('forbidden_endings', []))}.",
        "Allowed dares: clothing, sentence, scent, book_excerpt.",
        f"Mode: {ctx.mode}.",
        f"Setting hint: {snap.location} | {snap.season} | {snap.time_of_day} | {snap.weather}. Use at most one subtle reference.",
    ]
    if facts:
        lines.append("Useful facts (weave naturally if helpful): " + " | ".join([f["text"] for f in facts]))
    return "\n".join(lines)

def call_openai(messages: List[Dict[str, str]]) -> str:
    url = f"{settings.openai_base_url}/chat/completions"
    payload = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.4,
    }
    headers = build_openai_headers()
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# FastAPI app & endpoints
# ----------------------------

app = FastAPI(title="Dear Gentle MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dear-gentle.surpuissant.io"],  # adapte
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY")

    # 1) Append user message
    msgs = CONVERSATIONS.setdefault(req.session_id, [])
    user_mode = detect_mode(req.message, default_mode="short")
    msgs.append(Message(role="user", content=req.message, mode=user_mode))

    # 2) Build context package
    ctx, used_facets, used_mem_ids = build_context(
        user_id=req.user_id,
        session_id=req.session_id,
        user_text=req.message,
        mode=user_mode if user_mode != "instruction" else "short",  # instructions handled elsewhere
        snapshot_override=req.snapshot_override
    )

    # 3) System & user messages for LLM
    system_prompt = render_system_prompt(ctx)

    # For rewrite mode, we pass the conversation digest explicitly.
    user_payload = req.message
    if user_mode == "rewrite" and ctx.short_memory.get("summary"):
        # add compact digest of recent conversation
        history_lines = []
        for m in ctx.short_memory["recent_messages"]:
            history_lines.append(f"{m['role']}: {m['content']}")
        digest = "\n".join(history_lines[-12:])  # cap lines for brevity
        user_payload = f"++ REWRITE REQUEST\n\nConversation digest:\n{digest}\n\nUser says:\n{req.message}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload}
    ]

    # 4) Call LLM
    output = call_openai(messages)

    # 5) Persist assistant message
    CONVERSATIONS[req.session_id].append(Message(role="assistant", content=output, mode=user_mode))

    # 6) Mark facets & memory recency
    snap = SNAPSHOTS.get(req.session_id) or build_snapshot(req.session_id, None)
    snap = mark_facets_used(snap, req.session_id, used_facets)
    SNAPSHOTS[req.session_id] = snap
    rec = MEMORY_USE_RECENCY.setdefault(req.session_id, [])
    now = time.time()
    for mid in used_mem_ids:
        rec.append((mid, now))

    return ChatResponse(output=output, mode=user_mode, used_facets=used_facets, used_memory_ids=used_mem_ids)

# --- Helpers to seed preferences/memories for demo purposes ---

class SeedPrefRequest(BaseModel):
    user_id: str
    items: List[Preference]

@app.post("/api/seed/preferences")
def seed_prefs(req: SeedPrefRequest):
    d = PREFERENCES.setdefault(req.user_id, {})
    for it in req.items:
        d[it.key] = it.value
    return {"ok": True, "count": len(req.items)}

class SeedMemoryRequest(BaseModel):
    user_id: str
    texts: List[str]

@app.post("/api/seed/memories")
def seed_memories(req: SeedMemoryRequest):
    lst = MEMORIES.setdefault(req.user_id, [])
    for t in req.texts:
        lst.append(MemoryItem(
            id=str(uuid.uuid4()),
            user_id=req.user_id,
            text=t,
            embedding=encode_text_to_vec(t).tolist(),
            tags=[]
        ))
    return {"ok": True, "count": len(req.texts)}

class SetSnapshotRequest(BaseModel):
    session_id: str
    snapshot: Snapshot

@app.post("/api/snapshot/set")
def set_snapshot(req: SetSnapshotRequest):
    SNAPSHOTS[req.session_id] = req.snapshot
    return {"ok": True}