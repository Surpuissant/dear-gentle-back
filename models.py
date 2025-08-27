# /models.py
import time
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


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
    season: Optional[str] = None  # "summer", "winter", ...
    time_of_day: Optional[str] = None  # "morning","afternoon","evening","night"
    weather: Optional[Dict[str, str]] = None  # {"condition":"mild_evening","temperature_c":27}
    contextual_facets: List[str] = []
    last_mentioned_facets: List[Dict[str, str]] = []  # [{"facet":"lac d’Annecy","ts":"..."}]
    cultural_refs: List[str] = []  # user-provided cultural references
    selected_facet: Optional[str] = None  # chosen once per turn


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
    book_id: str
    session_id: str
    message: str
    snapshot_override: Optional[Snapshot] = None


class ChatResponse(BaseModel):
    output: str
    mode: str
    used_facets: List[str] = []
    used_memory_ids: List[str] = []
    chapter_id: Optional[str] = None


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
# (1) Book Chapter Data models
# ----------------------------

class Book(BaseModel):
    id: str
    user_id: str
    title: str
    outline: List[str] = []  # ordered list of beat/scene summaries
    themes: List[str] = []
    style: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())


class Chapter(BaseModel):
    id: str
    book_id: str
    index: int  # 1-based chapter number
    title: str
    content: str
    summary: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())
    model: Optional[str] = None


class ChapterVersion(BaseModel):
    id: str
    chapter_id: str
    parent_version_id: Optional[str] = None
    title: str
    content: str
    notes: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())


class ChapterGenRequest(BaseModel):
    user_id: str
    book_id: str
    chapter_index: int
    prompt: Optional[str] = None  # user guidance (tone, POV, constraints)
    use_outline: bool = True
    use_prev_chapters: int = 2  # how many previous chapters to include as compressed context
    snapshot_override: Optional[Snapshot] = None


class ChapterEditRequest(BaseModel):
    user_id: str
    chapter_id: str
    edit_instruction: str  # free text, e.g., "rends-le plus mystérieux, 10% plus court"
