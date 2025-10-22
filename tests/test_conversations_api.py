from pathlib import Path
import os
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import app  # noqa: E402
from models import Message  # noqa: E402


def make_message(role: str, content: str, mode: str | None = None) -> Message:
    return Message(role=role, content=content, mode=mode)


def test_get_conversation_filters_instructions():
    session_id = "sess-api"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "Bonjour", mode="conversation"),
        make_message("user", ">> secret", mode="instruction"),
        make_message("assistant", "Salut !", mode="conversation"),
    ]
    app.SUMMARIES[session_id] = "Résumé courant"

    client = TestClient(app.app)
    resp = client.get(f"/api/conversations/{session_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["summary"] == "Résumé courant"
    items = payload["items"]
    assert len(items) == 2
    assert all(msg["mode"] != "instruction" for msg in items)
    assert [msg["role"] for msg in items] == ["user", "assistant"]

    app.CONVERSATIONS.pop(session_id, None)
    app.SUMMARIES.pop(session_id, None)


def test_get_conversation_applies_limit():
    session_id = "sess-limit"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "A"),
        make_message("assistant", "B"),
        make_message("user", "C"),
    ]

    client = TestClient(app.app)
    resp = client.get(f"/api/conversations/{session_id}", params={"limit": 1})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["content"] == "C"

    # limit=0 disables the cap
    resp_all = client.get(f"/api/conversations/{session_id}", params={"limit": 0})
    assert resp_all.status_code == 200
    assert len(resp_all.json()["items"]) == 3

    app.CONVERSATIONS.pop(session_id, None)
