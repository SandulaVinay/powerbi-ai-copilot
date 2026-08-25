# Power BI AI Copilot — Retrieval Architecture v2

## Why v2 exists

The original pipeline treated each question as an independent retrieval task. That worked well for direct questions but weakened conversational follow-ups such as:

1. `What are the latest Power BI updates?`
2. `Is there any recent update regarding RLS?`
3. `What is OLS?`

The second and third questions depend on the first topic. v2 adds a conversation-understanding layer before routing and retrieval.

## v2 pipeline

```text
User
  |
  v
Conversation State
  |
  +--> Follow-up Detection
  |
  +--> Active Topic Extraction
  |
  +--> Intent / Entity / Freshness Planning
  |
  v
Contextual Query
  |
  v
Existing QueryRouter
  |
  +-------------------+--------------------+
  |                   |                    |
  v                   v                    v
Local RAG        Official Web RAG      Reject
  |                   |
  +---------+---------+
            v
      Evidence + LLM
            |
            v
      Grounded Answer
            |
            v
     Conversation State
```

## Design decisions

### 1. Deterministic conversation layer

`conversation.py` does not call an LLM. It keeps the latency and token cost low while solving the highest-value failure mode: ambiguous follow-ups.

For example:

```text
Previous:
What are the latest Power BI updates?

Current:
Is there any recent update regarding RLS?
```

becomes an internal retrieval query similar to:

```text
Power BI conversation context: What are the latest Power BI updates?
Current user question: Is there any recent update regarding RLS.
Answer the current question, using the previous context only when relevant.
```

This allows the existing temporal/web routing logic to see both the release context and the RLS entity.

### 2. Conversation-aware caching

The API caches using the contextualized retrieval query rather than the raw user sentence. This prevents a short question such as `What about RLS?` from being treated as identical across unrelated conversations.

Web answers remain protected by the existing cache policy and are not persistently cached.

### 3. Adaptive planning metadata

The conversation layer now exposes:

- `intent`
- `requires_web`
- `source_preference`
- `retrieval_strategy`
- `is_follow_up`
- `active_topic`
- `rewrite_reason`

The current retrieval engine remains backward-compatible; these fields provide a controlled path toward adaptive retrieval without replacing the working BM25/hybrid implementation in one risky change.

### 4. Official-source discipline

The existing `web_rag.py` remains responsible for official Microsoft-domain filtering and ranking. The new conversation layer does not weaken that rule.

## Performance strategy

The v2 conversation layer is in-process and deterministic:

- no extra LLM request
- no extra embedding generation
- no extra web request
- bounded conversation memory
- bounded context size

Therefore the expected latency overhead is negligible compared with web retrieval, reranking, and LLM generation.

## Regression cases

`tests/test_conversation.py` covers:

- latest-update → RLS follow-up
- latest-update → OLS follow-up
- standalone DAX question

## Next planned phases

### Phase 2 — Evidence confidence

Replace the current binary BM25/token-overlap evidence check with a calibrated confidence score using retrieval rank, semantic similarity, reranker score, source authority, and query freshness.

### Phase 3 — Structured release knowledge

Create a release-level knowledge layer with fields such as:

```json
{
  "release": "August 2026",
  "category": "Copilot and AI",
  "feature": "...",
  "security": {"RLS": true, "OLS": true},
  "status": "GA",
  "source": "Microsoft Learn"
}
```

This allows security/release questions to be answered from structured facts instead of relying entirely on chunk retrieval.

### Phase 4 — Adaptive retrieval

Use the planner to choose the cheapest retrieval strategy that can answer safely:

```text
Simple evergreen question -> BM25
Weak local evidence       -> Hybrid
Latest/current question   -> Official Web
Complex current research  -> Hybrid + Official Web
```

### Phase 5 — Evaluation harness

Track retrieval recall, citation accuracy, groundedness, follow-up accuracy, freshness, latency, and token cost against a fixed Power BI benchmark set.
