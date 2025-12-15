from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list[dict]
    mode: str | None = None

@app.get("/")
def index():
    return FileResponse("web/index.html")

@app.post("/api/chat")
def chat(req: ChatRequest):
    last_user = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    return {"reply": "Заглушка: ты написал(а): " + last_user}