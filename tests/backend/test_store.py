import time
from dataclasses import dataclass
from typing import List

import pytest

from app.store import InMemoryConversationStore


@dataclass
class ChatMessage:
    role: str
    content: str


def test_new_session_id_prefix_and_uniqueness():
    store = InMemoryConversationStore()
    a = store.new_session_id()
    b = store.new_session_id()
    assert a.startswith("sess_")
    assert b.startswith("sess_")
    assert a != b


def test_get_or_create_creates_when_none():
    store = InMemoryConversationStore()
    sid = store.get_or_create(None)
    assert sid.startswith("sess_")
    assert store.get_messages(sid) == []


def test_get_or_create_returns_existing():
    store = InMemoryConversationStore()
    sid = store.get_or_create(None)
    sid2 = store.get_or_create(sid)
    assert sid2 == sid


def test_set_messages_overwrites_and_get_returns_copy(monkeypatch):
    store = InMemoryConversationStore()

    base_time = 1000.0
    now = {"t": base_time}

    def fake_time():
        return now["t"]

    import app.store as mod
    monkeypatch.setattr(mod.time, "time", fake_time)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    sid = store.get_or_create(None)

    now["t"] = base_time + 1
    messages = [ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b")]
    store.set_messages(sid, messages)

    got = store.get_messages(sid)
    assert got == messages
    got.append(ChatMessage(role="user", content="x"))
    assert store.get_messages(sid) == messages


def test_append_creates_session_and_updates(monkeypatch):
    store = InMemoryConversationStore()

    base_time = 2000.0
    now = {"t": base_time}

    def fake_time():
        return now["t"]

    import app.store as mod
    monkeypatch.setattr(mod.time, "time", fake_time)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    sid = "sess_manual"

    now["t"] = base_time + 1
    store.append(sid, ChatMessage(role="user", content="hi"))
    assert store.get_messages(sid) == [ChatMessage(role="user", content="hi")]

    now["t"] = base_time + 2
    store.append(sid, ChatMessage(role="assistant", content="hello"))
    assert store.get_messages(sid) == [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]


def test_reset_removes_session(monkeypatch):
    store = InMemoryConversationStore()

    import app.store as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    sid = store.get_or_create(None)
    store.append(sid, ChatMessage(role="user", content="x"))
    assert store.get_messages(sid) != []
    store.reset(sid)
    assert store.get_messages(sid) == []


def test_gc_ttl_removes_old_sessions(monkeypatch):
    store = InMemoryConversationStore(ttl_seconds=10, max_sessions=100)

    base_time = 3000.0
    now = {"t": base_time}

    def fake_time():
        return now["t"]

    import app.store as mod
    monkeypatch.setattr(mod.time, "time", fake_time)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    sid1 = store.get_or_create(None)
    sid2 = store.get_or_create(None)

    now["t"] = base_time + 1
    store.append(sid1, ChatMessage(role="user", content="a"))

    now["t"] = base_time + 2
    store.append(sid2, ChatMessage(role="user", content="b"))

    now["t"] = base_time + 20
    store.get_or_create(None)

    assert store.get_messages(sid1) == []
    assert store.get_messages(sid2) == []


def test_gc_max_sessions_keeps_most_recent(monkeypatch):
    store = InMemoryConversationStore(ttl_seconds=10_000, max_sessions=2)

    base_time = 4000.0
    now = {"t": base_time}

    def fake_time():
        return now["t"]

    import app.store as mod
    monkeypatch.setattr(mod.time, "time", fake_time)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    sid_a = store.get_or_create(None)
    now["t"] = base_time + 1
    store.append(sid_a, ChatMessage(role="user", content="a"))

    sid_b = store.get_or_create(None)
    now["t"] = base_time + 2
    store.append(sid_b, ChatMessage(role="user", content="b"))

    sid_c = store.get_or_create(None)
    now["t"] = base_time + 3
    store.append(sid_c, ChatMessage(role="user", content="c"))

    existing_messages = {
        sid_a: store.get_messages(sid_a),
        sid_b: store.get_messages(sid_b),
        sid_c: store.get_messages(sid_c),
    }

    count_non_empty = sum(1 for v in existing_messages.values() if v != [])
    assert count_non_empty == 2

    assert store.get_messages(sid_c) == [ChatMessage(role="user", content="c")]
    assert store.get_messages(sid_a) == []