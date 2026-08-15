import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

WEB_SEARCH_ENABLED = True

WEB_SEARCH_MAX_RESULTS = 5

WEB_SEARCH_DEPTH = "basic"

WEB_CACHE_TTL_SECONDS = 3600

LOCAL_CACHE_TTL_SECONDS = 86400

MAX_COMPLETION_TOKENS = 800

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================



# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

STATIC_DIR = BASE_DIR / "static"

CHUNKS_FILE = DATA_DIR / "web_chunks_clean.json"

CACHE_FILE = DATA_DIR / "answer_cache.json"


# ============================================================
# GROQ / LLM
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# ============================================================
# RERANKER MODEL
# ============================================================

RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# RETRIEVAL
# ============================================================

BM25_TOP_K = 10

EMBEDDING_TOP_K = 10

RRF_TOP_K = 10

FINAL_TOP_K = 3

RRF_K = 60


# ============================================================
# CACHE
# ============================================================

CACHE_TTL_SECONDS = int(
    os.getenv(
        "CACHE_TTL_SECONDS",
        "604800"
    )
)

CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv(
        "CACHE_SIMILARITY_THRESHOLD",
        "0.94"
    )
)


# ============================================================
# RATE LIMIT
# ============================================================

MAX_QUESTIONS_PER_DAY = int(
    os.getenv(
        "MAX_QUESTIONS_PER_DAY",
        "30"
    )
)


# ============================================================
# LLM GENERATION
# ============================================================

MAX_COMPLETION_TOKENS = 800

TEMPERATURE = 0.0


# ============================================================
# VALIDATE SECRET
# ============================================================

if not GROQ_API_KEY:

    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Set it in the .env file or cloud environment."
    )
