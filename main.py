from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load env vars
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app init
app = FastAPI()

# PATCH: Add OPTIONS to all routes (CORS preflight fix)
@app.on_event("startup")
def ensure_options_allowed():
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.methods.update(["OPTIONS"])

# CORS middleware
origins = [
    "http://localhost:5173",
    "https://louaialsabbagh.tech"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Health check
@app.get("/ping")
def ping():
    return {"message": "pong"}

# Schema
class Question(BaseModel):
    message: str

# Louai base context
LOUAI_BASE_CONTEXT = """
You are Louai's personal assistant, answering as Louai himself.
Always speak clearly, professionally, and concisely.

Louai is a programming student based in Canada. He has hands-on experience with full-stack development, mainly using Python, FastAPI, PostgreSQL, React, and MongoDB.
He has built several real-world projects, including:
- An AI-powered job matching platform
- A digital library system with Stripe integration
- A fitness coaching app

Louai understands API development, authentication with JWT and OAuth, microservices using Docker/Kubernetes, and AI integration using OpenAI and NLP.

He communicates efficiently and focuses on facts, logic, and clarity.
Avoid filler language, exaggeration, jokes, or informal tone.
"""

# Keyword mapping
KEY_TOPICS = {
    "skills": ["skill", "tech", "technology", "stack", "tools", "programming", "languages", "framework"],
    "projects": ["project", "work", "experience", "build", "portfolio", "code", "app"],
    "personality": ["personality", "soft skills", "attitude", "work style", "values", "philosophy"],
    "goals": ["goal", "dream", "startup", "vision", "future", "ambition", "career"],
    "languages": ["language", "speak", "fluent", "communication", "multilingual"],
    "general": []
}

def detect_topic(message: str) -> str:
    msg = message.lower()
    for topic, keywords in KEY_TOPICS.items():
        if any(word in msg for word in keywords):
            return topic
    return "general"

def is_simple_question(message: str) -> bool:
    simple_keywords = ["what", "who", "where", "when", "why", "how", "your name", "skill", "language", "hello"]
    msg = message.lower()
    return len(msg.split()) <= 10 and any(kw in msg for kw in simple_keywords)

@app.post("/about-louai")
@limiter.limit("5/minute")
async def about_me(request: Request, q: Question):
    if not q.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    topic = detect_topic(q.message)
    simple = is_simple_question(q.message)

    length_instruction = (
        "Keep it short and direct (1-2 sentences)." if simple else "Provide a professional and concise answer."
    )

    if topic == "skills":
        prompt = f"""
The question is about Louai's technical skills.
{length_instruction}

Mention FastAPI, React, Python, JWT, Docker, CI/CD, and backend architecture.

Question: {q.message}
"""
    elif topic == "projects":
        prompt = f"""
The question is about Louai's project experience.
{length_instruction}

Refer to:
- AI job-matching platform
- Digital library (Stripe)
- Fitness app project

Keep it factual and structured.

Question: {q.message}
"""
    elif topic == "personality":
        prompt = f"""
The question is about Louai’s personal traits or mindset.
{length_instruction}

Mention his disciplined work ethic, logical thinking, and focus on intentional development.

Question: {q.message}
"""
    elif topic == "goals":
        prompt = f"""
The question is about Louai's future goals.
{length_instruction}

State his plan to launch a remote-first startup, grow as a backend engineer, and build software with purpose.

Question: {q.message}
"""
    elif topic == "languages":
        prompt = f"""
The question is about Louai’s language or communication skills.
{length_instruction}

Mention he's fluent in English, French, and Spanish, and communicates effectively across teams.

Question: {q.message}
"""
    else:
        prompt = f"""
The question is:
{q.message}

{length_instruction}

Provide a direct, informative, and professional answer as Louai.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": LOUAI_BASE_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return {"response": response.choices[0].message.content.strip()}

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong while processing your request.")
