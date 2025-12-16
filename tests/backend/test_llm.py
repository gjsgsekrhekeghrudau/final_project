from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest


@dataclass
class ChatMessage:
    role: str
    content: str


class DummyResponses:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.next_response: Any = None

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.next_response


class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()


class DummyRespWithText:
    def __init__(self, output_text: Optional[str], resp_id: Optional[str] = None):
        self.output_text = output_text
        self.id = resp_id

    def __str__(self):
        return f"DummyRespWithText(output_text={self.output_text!r}, id={self.id!r})"


class DummyRespNoOutputText:
    def __init__(self, resp_id: Optional[str] = None):
        self.id = resp_id

    def __str__(self):
        return "DummyRespNoOutputText()"


def import_llm_module(monkeypatch):
    import app.llm as mod
    return mod


def test_init_raises_if_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mod = import_llm_module(monkeypatch)
    with pytest.raises(RuntimeError):
        mod.OpenAIResponsesProvider(model="x", api_key=None)


def test_init_uses_explicit_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    mod = import_llm_module(monkeypatch)

    created = {"api_key": None}

    class OpenAIStub:
        def __init__(self, api_key):
            created["api_key"] = api_key

    monkeypatch.setattr(mod, "OpenAI", OpenAIStub)

    provider = mod.OpenAIResponsesProvider(model="m", api_key="explicit_key")
    assert provider.model == "m"
    assert created["api_key"] == "explicit_key"


def test_init_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    mod = import_llm_module(monkeypatch)

    created = {"api_key": None}

    class OpenAIStub:
        def __init__(self, api_key):
            created["api_key"] = api_key

    monkeypatch.setattr(mod, "OpenAI", OpenAIStub)

    provider = mod.OpenAIResponsesProvider(model="m", api_key=None)
    assert provider.model == "m"
    assert created["api_key"] == "env_key"


def test_generate_text_sends_messages_and_returns_output_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    mod = import_llm_module(monkeypatch)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    dummy_client = DummyClient()

    class OpenAIStub:
        def __init__(self, api_key):
            self.responses = dummy_client.responses

    monkeypatch.setattr(mod, "OpenAI", OpenAIStub)

    provider = mod.OpenAIResponsesProvider(model="test-model", api_key="k")

    dummy_client.responses.next_response = DummyRespWithText("hello", resp_id="resp_1")

    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="u"),
    ]

    text, meta = provider.generate_text(messages, temperature=0.9, max_output_tokens=123)

    assert text == "hello"
    assert meta == {"response_id": "resp_1", "model": "test-model"}

    call = dummy_client.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.9
    assert call["max_output_tokens"] == 123
    assert call["input"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_generate_text_falls_back_to_str_when_output_text_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    mod = import_llm_module(monkeypatch)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    dummy_client = DummyClient()

    class OpenAIStub:
        def __init__(self, api_key):
            self.responses = dummy_client.responses

    monkeypatch.setattr(mod, "OpenAI", OpenAIStub)

    provider = mod.OpenAIResponsesProvider(model="test-model", api_key="k")

    dummy_client.responses.next_response = DummyRespNoOutputText(resp_id="resp_2")

    messages = [ChatMessage(role="user", content="hi")]
    text, meta = provider.generate_text(messages)

    assert text == "DummyRespNoOutputText()"
    assert meta == {"response_id": "resp_2", "model": "test-model"}


def test_generate_text_passes_extra_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    mod = import_llm_module(monkeypatch)
    monkeypatch.setattr(mod, "ChatMessage", ChatMessage)

    dummy_client = DummyClient()

    class OpenAIStub:
        def __init__(self, api_key):
            self.responses = dummy_client.responses

    monkeypatch.setattr(mod, "OpenAI", OpenAIStub)

    provider = mod.OpenAIResponsesProvider(model="test-model", api_key="k")

    dummy_client.responses.next_response = DummyRespWithText("ok", resp_id="resp_3")

    messages = [ChatMessage(role="user", content="hi")]
    text, meta = provider.generate_text(
        messages,
        extra={"timeout": 10, "metadata": {"a": 1}},
    )

    assert text == "ok"
    assert meta == {"response_id": "resp_3", "model": "test-model"}

    call = dummy_client.responses.calls[0]
    assert call["timeout"] == 10
    assert call["metadata"] == {"a": 1}