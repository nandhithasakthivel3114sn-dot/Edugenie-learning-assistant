"""
EduGenie - Google Gemini Powered Learning Assistant
Backend: FastAPI

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload
    Then open http://127.0.0.1:8000
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

# ---------- 1. Configure Gemini API ----------
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")  # set as env variable, don't hardcode
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-flash-latest"  # fast + cheap; swap to gemini-1.5-pro for harder tasks
model = genai.GenerativeModel(MODEL_NAME)

# ---------- 2. FastAPI app setup ----------
app = FastAPI(title="EduGenie API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 3. Request models ----------
class QARequest(BaseModel):
    question: str


class SummaryRequest(BaseModel):
    text: str


class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5


# ---------- 4. Helper to call Gemini safely ----------
def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# ---------- 5. Feature 1: Q&A ----------
@app.post("/api/qa")
def question_answer(req: QARequest):
    prompt = f"Answer this student's question clearly and simply:\n\n{req.question}"
    answer = ask_gemini(prompt)
    return {"answer": answer}


# ---------- 6. Feature 2: Summarization ----------
@app.post("/api/summarize")
def summarize(req: SummaryRequest):
    prompt = f"Summarize the following text in 4-5 short bullet points:\n\n{req.text}"
    summary = ask_gemini(prompt)
    return {"summary": summary}


# ---------- 7. Feature 3: Quiz generation ----------
@app.post("/api/quiz")
def generate_quiz(req: QuizRequest):
    prompt = f"""
Create {req.num_questions} multiple-choice questions on the topic: {req.topic}.
Return ONLY valid JSON, no extra text, in this exact format:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]
"""
    raw = ask_gemini(prompt)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        quiz = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not parse quiz JSON from model output")
    return {"quiz": quiz}


# ---------- 8. Serve frontend ----------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))