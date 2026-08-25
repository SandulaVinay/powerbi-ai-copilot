from collections import defaultdict
from datetime import date
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import STATIC_DIR, CACHE_FILE, CACHE_TTL_SECONDS, MAX_QUESTIONS_PER_DAY
from cache import AnswerCache
from conversation import ConversationManager
from rag_engine import RAGEngine


app = FastAPI(
    title="Power BI Copilot",
    description="Production RAG assistant for Power BI documentation",
    version="2.0.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

print("=" * 70)
print("INITIALIZING POWER BI COPILOT")
print("=" * 70)
rag_engine = RAGEngine()
answer_cache = AnswerCache(CACHE_FILE, ttl_seconds=CACHE_TTL_SECONDS)
conversation_manager = ConversationManager()

rate_lock = Lock()
daily_usage = defaultdict(int)
daily_usage_date = date.today()


def check_rate_limit(client_id: str) -> bool:
    global daily_usage_date
    today = date.today()
    with rate_lock:
        if today != daily_usage_date:
            daily_usage.clear()
            daily_usage_date = today
        if daily_usage[client_id] >= MAX_QUESTIONS_PER_DAY:
            return False
        daily_usage[client_id] += 1
        return True


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    history: list[ChatTurn] = Field(default_factory=list)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "powerbi-copilot",
        "version": "2.0.0",
        "cache_entries": answer_cache.count(),
        "conversation_sessions": len(conversation_manager._sessions),
    }


@app.post("/api/chat")
def chat_endpoint(request: Request, response: Response, payload: ChatRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client_id = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Daily question limit reached. Please try again tomorrow.",
        )

    conversation_id = (
        payload.conversation_id
        or request.cookies.get("pbi_copilot_conversation_id")
        or str(uuid4())
    )
    response.set_cookie(
        key="pbi_copilot_conversation_id",
        value=conversation_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )

    history = [turn.model_dump() for turn in payload.history]
    state = conversation_manager.set_history(conversation_id, history)
    plan = conversation_manager.contextualize(question, state)
    retrieval_query = plan["query"]

    # Context-aware cache key: identical short questions can mean different
    # things in different conversations, so follow-ups use their rewritten query.
    cached = answer_cache.get(retrieval_query)
    if cached:
        conversation_manager.add_turn(conversation_id, "user", question)
        conversation_manager.add_turn(conversation_id, "assistant", cached["answer"])
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "cached": True,
            "conversation_id": conversation_id,
            "route": cached.get("route", "local"),
            "retrieval_type": cached.get("retrieval_type", "local_hybrid"),
            "retrieval_time": 0.0,
            "llm_time": 0.0,
            "total_time": 0.0,
            "tokens": {"input": 0, "output": 0},
            "conversation": {
                "is_follow_up": plan["is_follow_up"],
                "active_topic": plan["active_topic"],
                "rewrite_reason": plan["rewrite_reason"],
                "retrieval_query": retrieval_query,
            },
        }

    try:
        result = rag_engine.ask(retrieval_query)
    except Exception as error:
        print("\nRAG ERROR:", str(error))
        raise HTTPException(
            status_code=500,
            detail="The Power BI knowledge assistant encountered an internal error.",
        )

    cache_saved = answer_cache.set(
        retrieval_query,
        result["answer"],
        result.get("sources", []),
        route=result.get("route", "local"),
        retrieval_type=result.get("retrieval_type", "local_hybrid"),
    )

    conversation_manager.add_turn(conversation_id, "user", question)
    conversation_manager.add_turn(conversation_id, "assistant", result["answer"])

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "cached": False,
        "conversation_id": conversation_id,
        "route": result.get("route", "local"),
        "retrieval_type": result.get("retrieval_type", "unknown"),
        "retrieval_time": result.get("retrieval_time", 0.0),
        "llm_time": result.get("llm_time", 0.0),
        "total_time": result.get("total_time", 0.0),
        "cache_saved": cache_saved,
        "tokens": {
            "input": result.get("input_tokens", 0),
            "output": result.get("output_tokens", 0),
        },
        "conversation": {
            "is_follow_up": plan["is_follow_up"],
            "active_topic": plan["active_topic"],
            "rewrite_reason": plan["rewrite_reason"],
            "retrieval_query": retrieval_query,
        },
    }


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")
