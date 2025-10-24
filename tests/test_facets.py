from pathlib import Path
import os
import sys

from models import Message, Snapshot

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import app  # noqa: E402


def make_message(role: str, content: str, mode: str | None = None) -> Message:
    return Message(role=role, content=content, mode=mode)


def test_facets_ignore_instruction_and_author_messages():
    session_id = "facet-mix"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "Salut", mode="conversation"),
        make_message("user", ">> mode auteur", mode="instruction"),
        make_message("assistant", "Bonsoir", mode="conversation"),
    ]

    snapshot = Snapshot(time_of_day="sunset")

    snapshot = app.mark_facets_used(snapshot, session_id, [snapshot.time_of_day])
    assert snapshot.last_mentioned_facets[-1]["ts"] == "2"

    app.CONVERSATIONS[session_id].extend(
        [
            make_message("user", ">> encore", mode="instruction"),
            make_message("assistant", "Chapitre 1", mode="author"),
            make_message("assistant", "Toujours là", mode="conversation"),
            make_message("user", "Oui", mode="conversation"),
            make_message("assistant", "Encore moi", mode="conversation"),
        ]
    )

    allowed = app.schedule_facets(snapshot, session_id)
    assert snapshot.time_of_day in allowed

    app.CONVERSATIONS.pop(session_id, None)
    app.SNAPSHOTS.pop(session_id, None)
    app.SNAPSHOT_STATE.pop(session_id, None)
