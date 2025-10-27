"""Tests around the author payload construction."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OPENAI_API_KEY", "test-key")


from app import AUTHOR_HISTORY_LEGEND, _build_author_user_payload  # noqa: E402


def test_author_payload_labels_narrator_and_gentle():
    history = [
        {"role": "user", "content": "Salut"},
        {"role": "assistant", "content": "Bonsoir"},
    ]

    payload = _build_author_user_payload(history, "Écris un chapitre.")

    assert AUTHOR_HISTORY_LEGEND in payload
    assert "Narrateur·ice — Salut" in payload
    assert "Gentle — Bonsoir" in payload
    assert payload.strip().endswith("Consignes:\nÉcris un chapitre.")


def test_author_payload_trims_instruction():
    history: list[dict[str, str]] = []
    payload = _build_author_user_payload(history, "   Inspire-toi des montagnes.   ")

    assert payload.endswith("Consignes:\nInspire-toi des montagnes.")
