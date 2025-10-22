import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OPENAI_API_KEY", "test-key")


from app import mmr_select  # noqa: E402
from models import MemoryItem  # noqa: E402


def make_memory(mem_id: str, embedding: list[float]) -> MemoryItem:
    return MemoryItem(id=mem_id, user_id="u", text=mem_id, embedding=embedding)


def test_mmr_select_skips_recent_memories():
    query = np.array([1.0, 0.0], dtype=np.float32)
    recent = make_memory("recent", [1.0, 0.0])
    fresh = make_memory("fresh", [0.9, 0.1])

    selected = mmr_select(
        query_vec=query,
        candidates=[recent, fresh],
        k=2,
        lambda_mult=0.5,
        used_recent_ids=["recent"],
    )

    returned_ids = {m.id for m in selected}
    assert "recent" not in returned_ids
    assert "fresh" in returned_ids
