import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OPENAI_API_KEY", "test-key")


import app  # noqa: E402
from models import Book, Message  # noqa: E402
from style_packs import get_style_pack  # noqa: E402


def make_message(role: str, content: str, mode: str | None = None) -> Message:
    return Message(role=role, content=content, mode=mode)


def test_initial_situation_present_on_first_turn():
    user_id = "user-init"
    session_id = "sess-init"
    book_id = "book-init"

    pack = get_style_pack("henry")

    try:
        app.CONVERSATIONS[session_id] = [
            make_message("user", "Salut", mode="conversation"),
        ]
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []

        ctx, _, _ = app.build_context(
            user_id=user_id,
            session_id=session_id,
            book_id=book_id,
            user_text="Salut",
            mode="conversation",
            snapshot_override=None,
        )

        assert ctx.is_first_turn is True
        assert ctx.initial_situation == pack.initial_situation

        prompt = app.render_system_prompt_conversation(
            ctx,
            app.ConvRegister.brevity,
            user_id,
        )

        assert pack.initial_situation in prompt
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)


def test_initial_situation_skipped_after_first_turn():
    user_id = "user-follow"
    session_id = "sess-follow"
    book_id = "book-follow"

    pack = get_style_pack("henry")

    try:
        app.CONVERSATIONS[session_id] = [
            make_message("user", "Salut", mode="conversation"),
            make_message("assistant", "Bonsoir", mode="conversation"),
            make_message("user", "Deuxième message", mode="conversation"),
        ]
        app.BOOKS[book_id] = Book(id=book_id, user_id=user_id, title="Roman")
        app.USERS[user_id] = {"style_id": "henry"}
        app.MEMORIES[user_id] = []

        ctx, _, _ = app.build_context(
            user_id=user_id,
            session_id=session_id,
            book_id=book_id,
            user_text="Deuxième message",
            mode="conversation",
            snapshot_override=None,
        )

        assert ctx.is_first_turn is False
        assert ctx.initial_situation == pack.initial_situation

        prompt = app.render_system_prompt_conversation(
            ctx,
            app.ConvRegister.brevity,
            user_id,
        )

        assert pack.initial_situation not in prompt
    finally:
        app.CONVERSATIONS.pop(session_id, None)
        app.BOOKS.pop(book_id, None)
        app.USERS.pop(user_id, None)
        app.MEMORIES.pop(user_id, None)
        app.SUMMARIES.pop(session_id, None)
        app.MEMORY_USE_RECENCY.pop(session_id, None)
