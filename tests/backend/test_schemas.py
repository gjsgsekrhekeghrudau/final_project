import pytest
from pydantic import ValidationError

from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EvaluateRequest,
    EvaluateResponse,
)


def test_chat_message_valid_roles():
    m1 = ChatMessage(role="system", content="a")
    m2 = ChatMessage(role="user", content="b")
    m3 = ChatMessage(role="assistant", content="c")
    assert m1.role == "system"
    assert m2.role == "user"
    assert m3.role == "assistant"


def test_chat_message_invalid_role_raises():
    with pytest.raises(ValidationError):
        ChatMessage(role="admin", content="x")


def test_chat_request_defaults():
    req = ChatRequest()
    assert req.session_id is None
    assert req.mode == "interview_coach"
    assert req.messages == []
    assert req.meta == {}


def test_chat_request_parses_messages():
    req = ChatRequest(
        session_id="s1",
        mode="interview_coach",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        meta={"a": 1},
    )
    assert req.session_id == "s1"
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hi"
    assert req.messages[1].role == "assistant"
    assert req.meta == {"a": 1}


def test_chat_request_invalid_messages_type_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages="not a list")


def test_chat_response_defaults_meta():
    resp = ChatResponse(session_id="s1", reply="ok")
    assert resp.session_id == "s1"
    assert resp.reply == "ok"
    assert resp.meta == {}


def test_evaluate_request_requires_fields():
    r = EvaluateRequest(question="q", answer="a")
    assert r.question == "q"
    assert r.answer == "a"

    with pytest.raises(ValidationError):
        EvaluateRequest(question="q")


def test_evaluate_response_score_bounds():
    ok0 = EvaluateResponse(score=0, feedback="f", improved_answer="i")
    ok10 = EvaluateResponse(score=10, feedback="f", improved_answer="i")
    assert ok0.score == 0
    assert ok10.score == 10

    with pytest.raises(ValidationError):
        EvaluateResponse(score=-1, feedback="f", improved_answer="i")

    with pytest.raises(ValidationError):
        EvaluateResponse(score=11, feedback="f", improved_answer="i")


def test_evaluate_response_requires_strings():
    with pytest.raises(ValidationError):
        EvaluateResponse(score=5, feedback=123, improved_answer="i")

    with pytest.raises(ValidationError):
        EvaluateResponse(score=5, feedback="f", improved_answer=123)