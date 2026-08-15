# Vinay's Power BI Copilot

Power BI-focused RAG assistant built with FastAPI, BM25 retrieval, optional hybrid retrieval, Tavily web search, and Groq.

## Production / Render Free mode

The default retrieval mode is **BM25** so the service stays within Render's small-memory instance limits. Heavy `sentence-transformers` and PyTorch dependencies are kept in `requirements-local.txt` for local hybrid experiments.

Required environment variables:

- `GROQ_API_KEY`
- `TAVILY_API_KEY`

Optional:

- `RAG_MODE=bm25` (default, recommended for small hosts)
- `RAG_MODE=hybrid` (local use; requires `requirements-local.txt`)

## Run locally

```powershell
pip install -r requirements.txt
$env:RAG_MODE="bm25"
uvicorn app:app --host 0.0.0.0 --port 8000
```

For hybrid local retrieval:

```powershell
pip install -r requirements-local.txt
$env:RAG_MODE="hybrid"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Render

`render.yaml` configures a free Python web service with the correct port command, Python 3.12, and BM25 production mode. Add the two API keys in Render's Environment settings; they are intentionally marked `sync: false` and are never stored in Git.
