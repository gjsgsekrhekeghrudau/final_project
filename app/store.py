import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .schemas import ChatMessage


@dataclass
class SessionData:
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)


class InMemoryConversationStore:
    def __init__(self, ttl_seconds: int = 21600, max_sessions: int = 5000):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: Dict[str, SessionData] = {}

    def new_session_id(self) -> str:
        return "sess_" + uuid.uuid4().hex

    def get_or_create(self, session_id: Optional[str]) -> str:
        self._gc()
        if session_id and session_id in self._sessions:
            return session_id
        sid = self.new_session_id()
        self._sessions[sid] = SessionData()
        return sid

    def set_messages(self, session_id: str, messages: List[ChatMessage]) -> None:
        self._gc()
        self._sessions[session_id] = self._sessions.get(session_id, SessionData())
        self._sessions[session_id].messages = messages
        self._sessions[session_id].updated_at = time.time()

    def append(self, session_id: str, message: ChatMessage) -> None:
        self._gc()
        self._sessions[session_id] = self._sessions.get(session_id, SessionData())
        self._sessions[session_id].messages.append(message)
        self._sessions[session_id].updated_at = time.time()

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        self._gc()
        s = self._sessions.get(session_id)
        return list(s.messages) if s else []

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _gc(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if (now - s.updated_at) > self.ttl]
        for sid in dead:
            self._sessions.pop(sid, None)
        if len(self._sessions) > self.max_sessions:
            items: List[Tuple[str, SessionData]] = sorted(self._sessions.items(), key=lambda kv: kv[1].updated_at)
            for sid, _ in items[: len(self._sessions) - self.max_sessions]:
                self._sessions.pop(sid, None)
