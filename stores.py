# stores.py
# Centralize in-memory stores to avoid circular imports.

from typing import Dict, List

from models import InstructionOverride, Chapter, ChapterVersion

USERS: Dict[str, Dict[str, str]] = {}
INSTRUCTIONS: Dict[str, List[InstructionOverride]] = {}  # user_id -> overrides

CHAPTERS: Dict[str, Chapter] = {}
CHAPTER_VERSIONS: Dict[str, List[ChapterVersion]] = {}
CHAPTER_EMB: Dict[str, List[float]] = {}  # embedding per chapter content (for later retrieval)
