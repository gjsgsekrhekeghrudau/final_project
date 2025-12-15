from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    mode: str = "interview_coach"
    messages: List[ChatMessage] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    question: str
    answer: str


class EvaluateResponse(BaseModel):
    score: int = Field(ge=0, le=10)
    feedback: str
    improved_answer: str
