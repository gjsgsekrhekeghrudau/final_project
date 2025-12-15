import json
from typing import Any, Dict, List
from .schemas import ChatMessage, EvaluateResponse
from .llm import LLMProvider

SYSTEM_PROMPT = """Ты — ассистент для подготовки к собеседованиям.
Цели:
- снижать тревогу и вести диалог дружелюбно;
- задавать вопросы от простого к сложному;
- по запросу давать подсказку / эталонный ответ / оценку.

Правила:
- Если пользователь пишет "старт" — задай 1 вопрос и попроси ответить.
- Если пользователь просит "подсказку" — дай 3–5 пунктов, не раскрывая весь ответ.
- Если пользователь просит "эталонный ответ" — дай структурированный полный ответ.
- Если пользователь просит "оценить" — верни балл 0–10, объясни почему и предложи улучшенную версию.
"""


class InterviewCoachService:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def chat(self, user_messages: List[ChatMessage], meta: Dict[str, Any] | None = None) -> tuple[str, Dict[str, Any]]:
        full = [ChatMessage(role="system", content=SYSTEM_PROMPT), *user_messages]
        text, llm_meta = self.llm.generate_text(full, temperature=0.3, max_output_tokens=900)
        out_meta = {**(meta or {}), **llm_meta}
        return text, out_meta

    def evaluate(self, question: str, answer: str) -> EvaluateResponse:
        schema_hint = {"score": 0, "feedback": "string", "improved_answer": "string"}
        prompt = [
            ChatMessage(role="system", content="Отвечай только валидным JSON без лишнего текста."),
            ChatMessage(
                role="user",
                content=(
                    "Оцени ответ кандидата по 10-балльной шкале и дай фидбек.\n\n"
                    f"Вопрос:\n{question}\n\n"
                    f"Ответ:\n{answer}\n\n"
                    "Формат ответа — JSON:\n"
                    f"{json.dumps(schema_hint, ensure_ascii=False)}"
                ),
            ),
        ]
        text, _ = self.llm.generate_text(prompt, temperature=0.2, max_output_tokens=600)
        try:
            data = json.loads(text)
        except Exception:
            data = {"score": 5, "feedback": text.strip(), "improved_answer": "Определи суть → шаги/причины → пример → ограничения → итог."}
        return EvaluateResponse(**data)
