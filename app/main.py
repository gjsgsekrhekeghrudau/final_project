import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .schemas import ChatRequest, ChatResponse, ChatMessage, EvaluateRequest, EvaluateResponse
from .store import InMemoryConversationStore
from .llm import OpenAIResponsesProvider
from .interview import InterviewCoachService

load_dotenv("secrets/.env")

app = FastAPI(title="Interview Coach Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = InMemoryConversationStore()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = OpenAIResponsesProvider(model=MODEL)
coach = InterviewCoachService(llm)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    sid = store.get_or_create(req.session_id)
    store.set_messages(sid, req.messages)
    reply, meta = coach.chat(store.get_messages(sid), meta=req.meta)
    store.append(sid, ChatMessage(role="assistant", content=reply))
    return ChatResponse(session_id=sid, reply=reply, meta=meta)


@app.post("/api/evaluate", response_model=EvaluateResponse)
def api_evaluate(req: EvaluateRequest):
    return coach.evaluate(req.question, req.answer)


@app.post("/api/reset")
def api_reset(req: ChatRequest):
    sid = store.get_or_create(req.session_id)
    store.reset(sid)
    return {"ok": True, "session_id": sid}
