import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

client = OpenAI(
    api_key=openai_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "final_project",
    },
)

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list[dict]
    mode: str | None = None

@app.get("/")
def index():
    return FileResponse("web/index.html")

@app.post("/api/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model=openai_model,
        messages=req.messages,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    return {"reply": answer}