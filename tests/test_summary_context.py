import os
import sys
from pathlib import Path

from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OPENAI_API_KEY", "test-key")


import app
from app import build_chapter_context, maybe_refresh_summary
from models import Book, Chapter, Message


def make_message(role: str, content: str, mode: str | None = None) -> Message:
    return Message(role=role, content=content, mode=mode)


def test_maybe_refresh_summary_skips_with_few_turns(monkeypatch):
    session_id = "sess-skip"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "salut"),
        make_message("assistant", "bonjour"),
        make_message("user", "ça va ?"),
    ]

    calls: list = []

    def fake_chat(_payload):  # pragma: no cover - guard to detect unexpected calls
        calls.append(_payload)
        return "should not happen"

    monkeypatch.setattr(app, "call_openai_chat", fake_chat)

    maybe_refresh_summary(session_id)

    assert calls == []
    assert session_id not in app.SUMMARIES
    assert session_id not in app.SUMMARY_STATE

    app.CONVERSATIONS.pop(session_id, None)


def test_maybe_refresh_summary_updates_summary(monkeypatch):
    session_id = "sess-update"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "premier"),
        make_message("assistant", "réponse"),
        make_message("user", "deuxième"),
        make_message("assistant", "réponse"),
        make_message("user", "troisième"),
        make_message("assistant", "réponse"),
    ]

    captured_payloads: list = []

    def fake_chat(payload):
        captured_payloads.append(payload)
        return " Nouveau résumé "

    monkeypatch.setattr(app, "call_openai_chat", fake_chat)

    maybe_refresh_summary(session_id)

    assert len(captured_payloads) == 1
    assert app.SUMMARIES[session_id] == "Nouveau résumé"
    assert app.SUMMARY_STATE[session_id]["last_turn"] == 6

    app.CONVERSATIONS.pop(session_id, None)
    app.SUMMARIES.pop(session_id, None)
    app.SUMMARY_STATE.pop(session_id, None)


def test_maybe_refresh_summary_handles_http_exception(monkeypatch):
    session_id = "sess-error"
    app.CONVERSATIONS[session_id] = [
        make_message("user", "un"),
        make_message("assistant", "deux"),
        make_message("user", "trois"),
        make_message("assistant", "quatre"),
    ]

    def fake_chat(_payload):
        raise HTTPException(status_code=500, detail="boom")

    monkeypatch.setattr(app, "call_openai_chat", fake_chat)

    maybe_refresh_summary(session_id)

    assert session_id not in app.SUMMARIES
    assert app.SUMMARY_STATE[session_id]["last_turn"] == 4

    app.CONVERSATIONS.pop(session_id, None)
    app.SUMMARY_STATE.pop(session_id, None)


def test_build_chapter_context_prioritizes_recent_and_similar(monkeypatch):
    fake_embed = lambda text: [1.0, 0.0, 0.0]

    monkeypatch.setattr(app, "embed", fake_embed)
    monkeypatch.setattr(app, "CHAPTERS", {})
    monkeypatch.setattr(app, "CHAPTER_EMB", {})
    monkeypatch.setattr(app, "MEMORIES", {})
    monkeypatch.setattr(app, "SUMMARIES", {"sess": "Résumé courant"})

    book = Book(
        id="book1",
        user_id="user",
        title="Roman",
        themes=["amour"],
    )

    chapters = [
        Chapter(id=f"ch{idx}", book_id=book.id, index=idx, title=f"Chap {idx}", content=f"Texte {idx}")
        for idx in range(1, 5)
    ]

    embeddings = {
        "ch1": [1.0, 0.0, 0.0],
        "ch2": [0.8, 0.2, 0.0],
        "ch3": [0.0, 1.0, 0.0],
        "ch4": [0.5, 0.5, 0.0],
    }

    for ch in chapters:
        app.CHAPTERS[ch.id] = ch
        app.CHAPTER_EMB[ch.id] = embeddings[ch.id]

    ctx = build_chapter_context(
        user_id="user",
        book=book,
        chapter_index=5,
        use_prev_chapters=3,
        session_id="sess",
        author_instruction="Focus sur la mer",
    )

    prev_ctx = ctx["prev_chapters"]
    assert len(prev_ctx) == 3
    assert prev_ctx[0].startswith("Chapitre 1")
    assert prev_ctx[1].startswith("Chapitre 2")
    assert prev_ctx[2].startswith("Chapitre 4")
    assert all("Chapitre 3" not in entry for entry in prev_ctx)
    assert ctx["long_facts"] == ""
