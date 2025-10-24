"""
Auto-structured memory for Henry
--------------------------------
Drop-in module to add passive memory capture from user messages.

No new deps. Uses your existing utils.embed, cosine_sim, call_openai_chat, and Pydantic.

Main features
- Extract up to 3 memory candidates (preference / biography / constraint / people / commitment)
- Confidence scoring; auto-accept >= 0.80, else queue as pending for UI review
- Semantic dedupe against existing memories (cosine > 0.92)
- Cooldown to avoid re-capturing same thing repeatedly per session
- Simple decay (older, never-used items fade)

Integration points (see end of file for quick patch notes)
- maybe_autocapture(user_id, session_id, msg)
- REST helpers: list_pending, accept_pending, reject_pending
"""

from __future__ import annotations
import json
import time
import uuid
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from pydantic import BaseModel, Field
logger = logging.getLogger("app")

logger.warning("Starting auto_memory module")

# These will be provided by your app:
# - embed(text) -> List[float]
# - cosine_sim(a: np.ndarray, b: np.ndarray) -> float
# - call_openai_chat(messages) -> str
# - global stores: MEMORIES (user_id -> List[MemoryItem])
# You will pass them via lightweight injection on init() or assign after import.


# ---------- Injection hooks (filled by app on import) ----------
embed = None  # type: ignore
cosine_sim = None  # type: ignore
call_openai_chat = None  # type: ignore
MEMORIES: Dict[str, List] = {}  # type: ignore  # MemoryItem list from your models

# Optional, for per-session cooldowns
MEMORY_USE_RECENCY: Dict[str, List[Tuple[str, float]]] = {}  # session_id -> [(mem_id, ts)]

# ---------- Local stores ----------
PENDING_AUTOMEM: Dict[str, List["AutoMemCandidate"]] = {}  # user_id -> candidates (awaiting accept)
AUTOMEM_RECENCY: Dict[str, List[Tuple[str, float]]] = {}    # session_id -> [(fingerprint, ts)]


# ---------- Models ----------

KINDS = {"preference", "biography", "constraint", "people", "commitment"}

class AutoMemCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    text: str
    kind: str
    confidence: float = 0.0
    ts: float = Field(default_factory=lambda: time.time())

    def taglist(self) -> List[str]:
        # Use MemoryItem.tags to carry metadata without changing your schema
        return [f"kind:{self.kind}", f"conf:{self.confidence:.2f}", f"ts:{int(self.ts)}", "source:auto"]


# ---------- Config knobs ----------

CONF_AUTO_ACCEPT = 0.80
SIM_DUPLICATE = 0.92
SESSION_COOLDOWN_SEC = 60 * 15  # 15 minutes: do not capture near-identical fact twice quickly
MAX_PENDING_PER_USER = 30

EXTRACT_SYSTEM = (
    "Extract up to 2 memory candidates from the user's message. "
    "A useful memory is some piece of information about the user that is concrete, durable, like a real human memory. "
    "It can be a preference, a biographical fact, a constraint, a pet, an object, people in their life, or a commitment they made. "
    "Return strict JSON array of objects with fields: text, kind in "
    "{preference, biography, constraint, people, commitment}, confidence in [0,1]. "
    "Only include concrete, durable facts about the user or actionable commitments. "
    "Return [] if nothing useful."
)


# ---------- Core ----------

def _parse_json_safe(s: str) -> List[dict]:
    s = s.strip()
    # Try direct JSON first
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    # Fallback: find first bracketed JSON array
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(s[start:end + 1])
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def extract_candidates(user_id: str, msg: str) -> List[AutoMemCandidate]:
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": msg},
    ]
    raw = call_openai_chat(
        messages,
        temperature=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    logger.warning(f"Candidate extraction raw: {raw}")
    arr = _parse_json_safe(raw)
    out: List[AutoMemCandidate] = []
    for it in arr[:3]:
        text = (it.get("text") or "").strip()
        kind = (it.get("kind") or "").strip().lower()
        conf = float(it.get("confidence") or 0.0)
        if not text or kind not in KINDS:
            continue
        if conf <= 0:
            continue
        out.append(AutoMemCandidate(user_id=user_id, text=text, kind=kind, confidence=conf))
    return out


def _recent_fingerprint(s: str) -> str:
    # Lightweight canonical fingerprint for session-level cooldown
    return s.lower().strip()[:160]


def _session_cooldown_ok(session_id: str, text: str) -> bool:
    now = time.time()
    fp = _recent_fingerprint(text)
    lst = AUTOMEM_RECENCY.setdefault(session_id, [])
    # purge old
    AUTOMEM_RECENCY[session_id] = [(f, t) for (f, t) in lst if now - t < SESSION_COOLDOWN_SEC]
    return all(f != fp for f, _ in AUTOMEM_RECENCY[session_id])


def _mark_session_recent(session_id: str, text: str):
    AUTOMEM_RECENCY.setdefault(session_id, []).append((_recent_fingerprint(text), time.time()))


def _is_semantic_duplicate(user_id: str, text: str) -> bool:
    existing = MEMORIES.get(user_id, [])
    if not existing:
        return False
    try:
        q = np.array(embed(text), dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)
    except Exception:
        # Be safe: if we cannot embed, do not auto-store to avoid junk
        return True
    sims = []
    for m in existing:
        try:
            v = np.array(m.embedding, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-9)
            sims.append(float(cosine_sim(q, v)))
        except Exception:
            continue
    return bool(sims and max(sims) >= SIM_DUPLICATE)


def _store_as_memory(c: AutoMemCandidate) -> Optional[str]:
    # Converts AutoMemCandidate to your MemoryItem without changing your models
    # MemoryItem(id, user_id, text, embedding, tags)
    try:
        vec = embed(c.text)
    except Exception:
        return None
    item = type("MemoryItemProxy", (), {})()  # minimal duck-typed shim
    item.id = str(uuid.uuid4())
    item.user_id = c.user_id
    item.text = c.text
    item.embedding = vec
    item.tags = c.taglist()
    MEMORIES.setdefault(c.user_id, []).append(item)
    return item.id


# How this works:
# - Call maybe_autocapture(user_id, session_id, msg) after storing user message
# - It sends msg to LLM to extract up to 2 memory candidates
# - Each candidate is deduped against existing memories (cosine_sim > 0.92)
# - If confidence >= 0.80, it is auto-stored as MemoryItem
# - Else, it is queued in PENDING_AUTOMEM for user review
# - Session-level cooldown (30 min) to avoid re-capturing same fact repeatedly
# - Use list_pending, accept_pending, reject_pending for REST endpoints
def maybe_autocapture(user_id: str, session_id: str, msg: str) -> Dict:
    """Extract, dedupe, and store/pending. Returns summary for logging/telemetry."""
    if not _session_cooldown_ok(session_id, msg):
        return {"ok": True, "skipped": "cooldown"}

    cands = extract_candidates(user_id, msg)
    if not cands:
        return {"ok": True, "accepted": 0, "pending": 0}

    accepted, pending = 0, 0
    accepted_ids: List[str] = []
    for c in cands:
        if _is_semantic_duplicate(user_id, c.text):
            continue
        if c.confidence >= CONF_AUTO_ACCEPT:
            mid = _store_as_memory(c)
            if mid:
                accepted += 1
                accepted_ids.append(mid)
        else:
            lst = PENDING_AUTOMEM.setdefault(user_id, [])
            if len(lst) >= MAX_PENDING_PER_USER:
                lst.pop(0)
            lst.append(c)
            pending += 1
    _mark_session_recent(session_id, msg)

    logger.warning(
        "auto_mem_result",
        extra={
            "user_id": user_id,
            "session_id": session_id,
            "accepted": accepted,
            "accepted_ids": accepted_ids,
            "pending": pending,
            "candidates": [c.dict() for c in cands],
        },
    )

    return {"ok": True, "accepted": accepted, "accepted_ids": accepted_ids, "pending": pending}


# ---------- REST helpers (to wire into FastAPI) ----------

def list_pending(user_id: str) -> List[AutoMemCandidate]:
    return PENDING_AUTOMEM.get(user_id, [])


def accept_pending(user_id: str, ids: List[str]) -> Dict:
    items = PENDING_AUTOMEM.get(user_id, [])
    keep, moved = [], []
    for c in items:
        if c.id in ids:
            if not _is_semantic_duplicate(user_id, c.text):
                _store_as_memory(c)
                moved.append(c.id)
        else:
            keep.append(c)
    PENDING_AUTOMEM[user_id] = keep
    return {"ok": True, "accepted": len(moved), "accepted_ids": moved}


def reject_pending(user_id: str, ids: List[str]) -> Dict:
    items = PENDING_AUTOMEM.get(user_id, [])
    keep = [c for c in items if c.id not in ids]
    PENDING_AUTOMEM[user_id] = keep
    return {"ok": True, "deleted": len(items) - len(keep)}


# ---------- Quick integration notes ----------
"""
In app.py:

1) Import and wire hooks after your utils are available:

    from auto_memory import maybe_autocapture, list_pending, accept_pending, reject_pending
    import auto_memory as autom
    autom.embed = embed
    autom.cosine_sim = cosine_sim
    autom.call_openai_chat = call_openai_chat
    autom.MEMORIES = MEMORIES
    autom.MEMORY_USE_RECENCY = MEMORY_USE_RECENCY

2) Call maybe_autocapture right after persisting the user message (and before build_context) in /api/chat and /api/chat+chapters:

    CONVERSATIONS.setdefault(req.session_id, []).append(Message(...))
    try:
        autom.maybe_autocapture(req.user_id, req.session_id, raw_user_text)
    except Exception:
        logger.warning("auto-mem capture failed", exc_info=True)

3) Add endpoints (example):

    @app.get("/api/memories/auto/pending")
    def get_auto_pending(user_id: str):
        return {"ok": True, "items": [c.dict() for c in autom.list_pending(user_id)]}

    @app.post("/api/memories/auto/accept")
    def post_auto_accept(user_id: str = Body(...), ids: List[str] = Body(...)):
        return autom.accept_pending(user_id, ids)

    @app.post("/api/memories/auto/reject")
    def post_auto_reject(user_id: str = Body(...), ids: List[str] = Body(...)):
        return autom.reject_pending(user_id, ids)

4) Telemetry: log the return of maybe_autocapture() under an OTel span / structlog for visibility.

5) Tuning: adjust CONF_AUTO_ACCEPT and SIM_DUPLICATE based on offline eval.
"""
