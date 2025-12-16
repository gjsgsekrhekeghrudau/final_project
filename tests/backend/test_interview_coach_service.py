import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest

from app.interview import InterviewCoachService, SYSTEM_PROMPT


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class EvaluateResponse:
    score: int
    feedback: str
    improved_answer: str


class FakeLLM:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.next_text: str = ""
        self.next_meta: Dict[str, Any] = {}

    def set_next(self, text: str, meta: Dict[str, Any] | None = None):
        self.next_text = text
        self.next_meta = meta or {}

    def generate_text(self, messages, temperature: float, max_output_tokens: int) -> Tuple[str, Dict[str, Any]]:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.next_text, dict(self.next_meta)


def test_chat_adds_system_prompt_and_calls_llm(monkeypatch):
    fake_llm = FakeLLM()
    fake_llm.set_next("Ответ от модели", {"provider": "fake"})

    import app.interview as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    service = InterviewCoachService(llm=fake_llm)

    user_messages = [ChatMessage(role="user", content="старт")]
    text, out_meta = service.chat(user_messages, meta={"session_id": "123"})

    assert text == "Ответ от модели"
    assert out_meta == {"session_id": "123", "provider": "fake"}

    call = fake_llm.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_output_tokens"] == 900

    messages = call["messages"]
    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1:] == user_messages


def test_chat_meta_merge_priority_llm_over_meta(monkeypatch):
    fake_llm = FakeLLM()
    fake_llm.set_next("ok", {"trace_id": "from_llm", "same": 2})

    import app.interview as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    service = InterviewCoachService(llm=fake_llm)

    _, out_meta = service.chat(
        [ChatMessage(role="user", content="hi")],
        meta={"same": 1, "client": "web"},
    )

    assert out_meta == {"same": 2, "client": "web", "trace_id": "from_llm"}


def test_evaluate_parses_valid_json(monkeypatch):
    fake_llm = FakeLLM()
    fake_llm.set_next(json.dumps({"score": 9, "feedback": "хорошо", "improved_answer": "ещё лучше"}))

    import app.interview as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)
    monkeypatch.setattr(mod, "EvaluateResponse", EvaluateResponse)

    service = InterviewCoachService(llm=fake_llm)

    result = service.evaluate("Что такое REST?", "Это стиль архитектуры.")

    assert result == EvaluateResponse(
        score=9,
        feedback="хорошо",
        improved_answer="ещё лучше",
    )

    call = fake_llm.calls[0]
    assert call["temperature"] == 0.2
    assert call["max_output_tokens"] == 600

    messages = call["messages"]
    assert messages[0].role == "system"
    assert "валидным JSON" in messages[0].content


def test_evaluate_fallback_when_not_json(monkeypatch):
    fake_llm = FakeLLM()
    fake_llm.set_next("Это не JSON, просто текст")

    import app.interview as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)
    monkeypatch.setattr(mod, "EvaluateResponse", EvaluateResponse)

    service = InterviewCoachService(llm=fake_llm)

    result = service.evaluate("Вопрос", "Ответ")

    assert result.score == 5
    assert "Это не JSON" in result.feedback
    assert "Определи суть" in result.improved_answer


def test_evaluate_invalid_json_missing_fields_raises(monkeypatch):
    fake_llm = FakeLLM()
    fake_llm.set_next(json.dumps({"score": 3}))

    import app.interview as mod
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)
    monkeypatch.setattr(mod, "EvaluateResponse", EvaluateResponse)

    service = InterviewCoachService(llm=fake_llm)

    with pytest.raises(TypeError):
        service.evaluate("Вопрос", "Ответ")