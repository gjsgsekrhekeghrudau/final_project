import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from .schemas import ChatMessage


class LLMProvider(ABC):
    @abstractmethod
    def generate_text(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 800,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class OpenAIResponsesProvider(LLMProvider):
    def __init__(self, model: str = "Free_GPT_KEY", api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_text(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 800,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        input_payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.responses.create(
            model=self.model,
            input=input_payload,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **(extra or {}),
        )
        text = getattr(resp, "output_text", None)
        if not text:
            text = str(resp)
        meta = {"response_id": getattr(resp, "id", None), "model": self.model}
        return text, meta
