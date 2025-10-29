import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import app  # noqa: E402
from models import Book, ChatRequest, Message  # noqa: E402
from routes import core as core_routes  # noqa: E402


def test_chat_allows_gentle_to_initiate(monkeypatch):
    user_id = "user-gentle-start"
    session_id = "sess-gentle-start"
    book_id = "book-gentle-start"

    captured = {}

    def fake_call(messages, **_):
        captured["messages"] = messages
        return "Premiers mots du gentle"

    monkeypatch.setattr(core_routes, "call_openai_chat", fake_call)
    monkeypatch.setattr(core_routes.autom, "maybe_autocapture", lambda *args, **kwargs: None)

    try:
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []
        app.CONVERSATIONS[session_id] = []

        req = ChatRequest(
            user_id=user_id,
            book_id=book_id,
            session_id=session_id,
            message="   ",
        )

        resp = core_routes.chat(req)

        assert resp.output == "Premiers mots du gentle"

        convo = app.CONVERSATIONS[session_id]
        assert len(convo) == 1
        assert convo[0].role == "assistant"
        assert convo[0].content == "Premiers mots du gentle"

        user_payloads = [m for m in captured.get("messages", []) if m["role"] == "user"]
        assert user_payloads, "expected a user payload sent to the model"
        assert "Initie la conversation" in user_payloads[-1]["content"]
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SNAPSHOTS.pop(session_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)
        app.SNAPSHOT_STATE.pop(session_id, None)


def test_chat_ignores_empty_followup_message(monkeypatch):
    user_id = "user-empty-followup"
    session_id = "sess-empty-followup"
    book_id = "book-empty-followup"

    def fail_call(*args, **kwargs):
        raise AssertionError("model should not be called for empty messages")

    def fail_autocapture(*args, **kwargs):
        raise AssertionError("autocapture should not run for empty messages")

    monkeypatch.setattr(core_routes, "call_openai_chat", fail_call)
    monkeypatch.setattr(core_routes.autom, "maybe_autocapture", fail_autocapture)

    try:
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []
        app.CONVERSATIONS[session_id] = [
            Message(role="assistant", content="Bonjour", mode="conversation")
        ]

        req = ChatRequest(
            user_id=user_id,
            book_id=book_id,
            session_id=session_id,
            message="   ",
        )

        resp = core_routes.chat(req)

        assert resp.output == ""
        assert resp.mode == "conversation"

        convo = app.CONVERSATIONS[session_id]
        assert len(convo) == 1
        assert convo[0].role == "assistant"
        assert convo[0].content == "Bonjour"
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SNAPSHOTS.pop(session_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)
        app.SNAPSHOT_STATE.pop(session_id, None)
