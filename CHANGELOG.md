# Changelog

## 2026-08-25 — Retrieval Architecture v2

### Added

- Conversation-aware retrieval context.
- Follow-up detection for questions such as `What about RLS?`.
- Active-topic extraction from recent user turns.
- Deterministic intent/freshness/retrieval planning metadata.
- Context-aware cache keys for follow-up questions.
- Conversation ID persistence through an HTTP-only cookie.
- Optional conversation history in `/api/chat` requests.
- Regression tests for RLS/OLS follow-up behavior.
- `ARCHITECTURE.md` documenting the v2 design and next phases.

### Preserved

- Existing BM25 retrieval.
- Existing hybrid embedding + reranker retrieval.
- Existing official Microsoft-domain web filtering.
- Existing web freshness behavior.
- Existing protection against persistent caching of live-web answers.

### Important behavior change

A conversation such as:

```text
What are the latest Power BI updates?
Is there any recent update regarding RLS?
What is OLS?
```

now carries the previous Power BI release context into retrieval instead of treating each message as an isolated query.
