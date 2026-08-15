from collections import defaultdict
from datetime import date
from threading import Lock

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse

from pydantic import BaseModel


from config import (
    STATIC_DIR,
    CACHE_FILE,
    CACHE_TTL_SECONDS,
    MAX_QUESTIONS_PER_DAY
)

from cache import AnswerCache

from rag_engine import RAGEngine


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Power BI Copilot",
    description=(
        "Production RAG assistant for Power BI documentation"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [

    "http://localhost:3000",

    "http://localhost:5173",

    "http://localhost:8000",

]


app.add_middleware(

    CORSMiddleware,

    allow_origins=ALLOWED_ORIGINS,

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "*"
    ]

)


# ============================================================
# RAG ENGINE
# ============================================================

print()
print("=" * 70)
print("INITIALIZING POWER BI COPILOT")
print("=" * 70)

rag_engine = RAGEngine()


# ============================================================
# ANSWER CACHE
# ============================================================

answer_cache = AnswerCache(

    CACHE_FILE,

    ttl_seconds=
        CACHE_TTL_SECONDS

)


# ============================================================
# SIMPLE DAILY RATE LIMITER
# ============================================================

rate_lock = Lock()

daily_usage = defaultdict(int)

daily_usage_date = date.today()


def check_rate_limit(
    client_id
):

    global daily_usage_date

    today = date.today()

    with rate_lock:

        # ----------------------------------------------------
        # Reset counters when the date changes
        # ----------------------------------------------------

        if today != daily_usage_date:

            daily_usage.clear()

            daily_usage_date = today

        # ----------------------------------------------------
        # Check limit
        # ----------------------------------------------------

        if (

            daily_usage[
                client_id
            ]

            >=

            MAX_QUESTIONS_PER_DAY

        ):

            return False

        # ----------------------------------------------------
        # Count request
        # ----------------------------------------------------

        daily_usage[
            client_id
        ] += 1

        return True


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "service":
            "powerbi-copilot",

        "cache_entries":
            answer_cache.count(),

    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
def chat_endpoint(
    request: Request,
    payload: ChatRequest
):

    question = (
        payload.question
        .strip()
    )

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not question:

        raise HTTPException(

            status_code=400,

            detail=(
                "Question cannot be empty."
            )

        )

    # ========================================================
    # IDENTIFY CLIENT
    #
    # Temporary development identifier.
    #
    # In production this should eventually become an
    # authenticated user ID or a properly designed
    # rate-limit identity.
    # ========================================================

    if request.client:

        client_id = (
            request.client.host
        )

    else:

        client_id = "unknown"

    # ========================================================
    # RATE LIMIT
    # ========================================================

    if not check_rate_limit(
        client_id
    ):

        raise HTTPException(

            status_code=429,

            detail=(
                "Daily question limit reached. "
                "Please try again tomorrow."
            )

        )

    # ========================================================
    # CACHE LOOKUP
    #
    # The AnswerCache itself now ensures that only valid
    # local RAG answers can be returned from persistent cache.
    #
    # Web answers are never persisted.
    # Rejected answers are never persisted.
    # Failed/unknown answers are never persisted.
    # ========================================================

    cached = answer_cache.get(
        question
    )

    if cached:

        return {

            "answer":
                cached[
                    "answer"
                ],

            "sources":
                cached[
                    "sources"
                ],

            "cached":
                True,

            "route":
                cached.get(
                    "route",
                    "local"
                ),

            "retrieval_type":
                cached.get(
                    "retrieval_type",
                    "local_hybrid"
                ),

            "retrieval_time":
                0.0,

            "llm_time":
                0.0,

            "total_time":
                0.0,

            "tokens": {

                "input":
                    0,

                "output":
                    0

            }

        }

    # ========================================================
    # RAG PIPELINE
    # ========================================================

    try:

        result = rag_engine.ask(
            question
        )

    except Exception as error:

        print(
            "\nRAG ERROR:"
        )

        print(
            str(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "The Power BI knowledge assistant "
                "encountered an internal error."
            )

        )

    # ========================================================
    # CACHE POLICY
    #
    # IMPORTANT:
    #
    # We now explicitly pass route and retrieval_type.
    #
    # LOCAL:
    #     Successful grounded answers can be cached.
    #
    # WEB:
    #     Never cached.
    #
    # REJECT:
    #     Never cached.
    #
    # UNKNOWN / FAILED:
    #     Never cached.
    #
    # The AnswerCache performs the final safety check.
    # ========================================================

    cache_saved = answer_cache.set(

        question,

        result[
            "answer"
        ],

        result.get(
            "sources",
            []
        ),

        route=result.get(
            "route",
            "local"
        ),

        retrieval_type=result.get(
            "retrieval_type",
            "local_hybrid"
        )

    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "answer":
            result[
                "answer"
            ],

        "sources":
            result.get(
                "sources",
                []
            ),

        "cached":
            False,

        "route":
            result.get(
                "route",
                "local"
            ),

        "retrieval_type":
            result.get(
                "retrieval_type",
                "unknown"
            ),

        "retrieval_time":
            result.get(
                "retrieval_time",
                0.0
            ),

        "llm_time":
            result.get(
                "llm_time",
                0.0
            ),

        "total_time":
            result.get(
                "total_time",
                0.0
            ),

        "cache_saved":
            cache_saved,

        "tokens": {

            "input":
                result.get(
                    "input_tokens",
                    0
                ),

            "output":
                result.get(
                    "output_tokens",
                    0
                )

        }

    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )