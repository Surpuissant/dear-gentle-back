"""Tests ensuring narrator/author exchanges stay private from the gentleman."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OPENAI_API_KEY", "test-key")


import app  # noqa: E402
from models import Book, Message  # noqa: E402


def make_message(role: str, content: str, mode: str | None = None) -> Message:
    return Message(role=role, content=content, mode=mode)


def test_build_context_hides_author_dialogues_from_gentle():
    user_id = "user-privacy"
    session_id = "sess-privacy"
    book_id = "book-privacy"

    try:
        app.CONVERSATIONS[session_id] = [
            make_message("user", "Salut", mode="conversation"),
            make_message("assistant", "Bonsoir", mode="conversation"),
            make_message("user", "::author Écris-moi un chapitre", mode="author"),
            make_message("assistant", "Chapitre secret", mode="author"),
            make_message("user", "Réécris ceci", mode="rewrite"),
            make_message("assistant", "Version réécrite", mode="rewrite"),
        ]
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []

        ctx, _, _ = app.build_context(
            user_id=user_id,
            session_id=session_id,
            book_id=book_id,
            user_text="Prochaine réponse",
            mode="conversation",
            snapshot_override=None,
        )

        recent = ctx.short_memory.get("recent_messages", [])
        contents = [msg["content"] for msg in recent]

        # Only the genuine conversation turns should remain.
        assert contents == ["Salut", "Bonsoir"]
        assert ctx.is_first_turn is False
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.SNAPSHOT_STATE.pop(session_id, None)
        app.SNAPSHOTS.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)


def test_build_context_ignores_narrator_whispers():
    user_id = "user-narrator"
    session_id = "sess-narrator"
    book_id = "book-narrator"

    try:
        app.CONVERSATIONS[session_id] = [
            make_message("user", "Salut", mode="conversation"),
            make_message("narrator", "(voix off) secret", mode="conversation"),
            make_message("assistant", "Bonsoir", mode="conversation"),
            make_message("narrator", "(voix off) un autre secret", mode="conversation"),
        ]
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []

        ctx, _, _ = app.build_context(
            user_id=user_id,
            session_id=session_id,
            book_id=book_id,
            user_text="Prochaine réplique",
            mode="conversation",
            snapshot_override=None,
        )

        recent = ctx.short_memory.get("recent_messages", [])
        roles = [msg["role"] for msg in recent]
        contents = [msg["content"] for msg in recent]

        assert roles == ["user", "assistant"]
        assert contents == ["Salut", "Bonsoir"]
        assert ctx.is_first_turn is False
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.SNAPSHOT_STATE.pop(session_id, None)
        app.SNAPSHOTS.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)

