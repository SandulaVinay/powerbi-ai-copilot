import json
import os
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from groq import Groq

from config import (
    CHUNKS_FILE,
    BM25_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
)
from query_router import QueryRouter
from web_rag import WebRAG


class RAGEngine:
    """Power BI RAG engine with a memory-safe production mode.

    RAG_MODE=bm25   -> BM25 + live Web RAG + Groq. Designed for small hosts.
    RAG_MODE=hybrid -> BM25 + embeddings + cross-encoder reranker for local use.

    Local retrieval has a domain-safe live-web fallback: if a Power BI / Fabric /
    DAX / BI question gets weak or incomplete local evidence, it is sent to WebRAG.
    Unrelated questions are never sent to the web fallback.
    """

    LOCAL_MIN_BM25_SCORE = 1.0
    LOCAL_MIN_TOKEN_OVERLAP = 2

    def __init__(self):
        print("=" * 70)
        print("POWER BI RAG ENGINE")
        print("=" * 70)

        self.mode = os.getenv("RAG_MODE", "hybrid").strip().lower()
        if self.mode not in {"bm25", "hybrid"}:
            self.mode = "bm25"

        print(f"Retrieval mode : {self.mode}")

        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            raw_chunks = json.load(file)

        unique_chunks = {}
        for chunk in raw_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                unique_chunks[chunk_id] = chunk

        self.chunks = list(unique_chunks.values())
        self.documents = [self.normalize(c.get("text", "")) for c in self.chunks]
        self.dax_functions = self._load_dax_function_catalog()

        print(f"Chunks loaded    : {len(self.chunks)}")
        print(
            "Unique documents : "
            f"{len({c.get('document_id') for c in self.chunks if c.get('document_id')})}"
        )
        print(f"DAX functions loaded : {len(self.dax_functions)}")

        print("\nBuilding BM25 index...")
        self.bm25 = BM25Okapi([self.tokenize(t) for t in self.documents])
        print("BM25 ready.")

        self.embedding_model = None
        self.document_embeddings = None
        self.reranker = None

        if self.mode == "hybrid":
            self._load_hybrid_models()
        else:
            print("\nMemory-safe production mode: BM25 only.")
            print("Skipping SentenceTransformer and CrossEncoder models.")

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        print("\nInitializing cloud LLM...")
        self.groq = Groq(api_key=GROQ_API_KEY)
        print(f"LLM model : {LLM_MODEL}")

        self.query_router = QueryRouter()
        print("Query router ready.")

        self.web_rag = None
        try:
            self.web_rag = WebRAG()
            print("Live web RAG ready.")
        except Exception as exc:
            print("WARNING: Live web RAG could not be initialized.")
            print(f"Reason: {exc}")

        print("\nRAG engine initialization complete.")
        print("=" * 70)

    @staticmethod
    def _load_dax_function_catalog():
        """Load the structured DAX catalog without making it a hard-coded router list."""
        catalog_path = os.path.join(
            os.path.dirname(CHUNKS_FILE),
            "dax_function_catalog.json",
        )
        try:
            with open(catalog_path, "r", encoding="utf-8") as file:
                catalog = json.load(file)
            functions = set()
            for names in catalog.get("categories", {}).values():
                functions.update(str(name).upper() for name in names)
            return functions
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return set()

    def _is_implicit_dax_question(self, question):
        """Recognize short DAX questions even when the user never says 'DAX'."""
        q = str(question).lower().strip()
        if not self.dax_functions:
            return False

        matched = [
            fn for fn in self.dax_functions
            if re.search(r"\b" + re.escape(fn.lower()) + r"\b", q)
        ]
        if not matched:
            return False

        has_question_signal = bool(
            re.search(
                r"\bvs\.?\b|\bversus\b|\bdifference\b|\bcompare\b|\bexplain\b|"
                r"\bwhat\s+is\b|\bhow\s+does\b|\bwhen\s+should\b|\bfunction(s)?\b",
                q,
            )
        )
        return has_question_signal or len(matched) >= 2

    def _load_hybrid_models(self):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from config import EMBEDDING_MODEL, RERANKER_MODEL

        print("\nLoading embedding model:")
        print(EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print("\nCreating document embeddings...")
        started = time.perf_counter()
        self.document_embeddings = self.embedding_model.encode(
            self.documents,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"Embedding creation time : {time.perf_counter() - started:.3f}s")

        print("\nLoading reranker:")
        print(RERANKER_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print("Reranker ready.")

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+", str(text).lower())

    @staticmethod
    def normalize(text):
        return re.sub(r"\s+", " ", str(text)).strip()

    def _local_bm25(self, question):
        scores = self.bm25.get_scores(self.tokenize(question))
        indices = np.argsort(scores)[::-1][:BM25_TOP_K]
        selected = []
        seen_docs = set()

        for raw_idx in indices:
            idx = int(raw_idx)
            doc_id = self.chunks[idx].get("document_id")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            selected.append((idx, float(scores[idx])))
            if len(selected) >= FINAL_TOP_K:
                break

        sources = []
        for rank, (idx, score) in enumerate(selected, 1):
            chunk = self.chunks[idx]
            sources.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "title": chunk.get("title", "Microsoft documentation"),
                "url": chunk.get("url", ""),
                "text": chunk.get("text", ""),
                "score": score,
                "source": "local",
            })
        return sources

    def _local_hybrid(self, question):
        from config import EMBEDDING_TOP_K, RRF_TOP_K

        bm_scores = self.bm25.get_scores(self.tokenize(question))
        bm_indices = np.argsort(bm_scores)[::-1][:BM25_TOP_K]
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
        )
        emb_scores = self.document_embeddings @ query_embedding
        emb_indices = np.argsort(emb_scores)[::-1][:EMBEDDING_TOP_K]

        rrf = {}
        for rank, idx in enumerate(bm_indices, 1):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(emb_indices, 1):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        candidates = [
            idx for idx, _ in sorted(
                rrf.items(), key=lambda x: x[1], reverse=True
            )[:RRF_TOP_K]
        ]
        pairs = [[question, self.documents[idx]] for idx in candidates]
        rerank_scores = self.reranker.predict(pairs)

        combined = [
            (idx, rrf[idx], float(score))
            for idx, score in zip(candidates, rerank_scores)
        ]
        combined.sort(key=lambda x: (x[1], x[2]), reverse=True)

        selected = []
        seen_docs = set()
        for idx, rrf_score, rerank_score in combined:
            doc_id = self.chunks[idx].get("document_id")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            selected.append((idx, rrf_score, rerank_score))
            if len(selected) >= FINAL_TOP_K:
                break

        return [
            {
                "rank": rank,
                "chunk_id": self.chunks[idx].get("chunk_id", ""),
                "document_id": self.chunks[idx].get("document_id", ""),
                "title": self.chunks[idx].get("title", "Microsoft documentation"),
                "url": self.chunks[idx].get("url", ""),
                "text": self.chunks[idx].get("text", ""),
                "score": float(rrf_score + rerank_score),
                "source": "local",
            }
            for rank, (idx, rrf_score, rerank_score) in enumerate(selected, 1)
        ]

    @staticmethod
    def _web_sources(result):
        sources = []
        for rank, item in enumerate(result.get("results", []), 1):
            sources.append({
                "rank": rank,
                "title": item.get("title", "Web source"),
                "url": item.get("url", ""),
                "text": (
                    item.get("content")
                    or item.get("snippet")
                    or item.get("description")
                    or ""
                ),
                "source": "web",
            })
        return sources

    @classmethod
    def _local_evidence_quality(cls, question, sources):
        """Decide whether local evidence is strong enough to answer safely."""
        if not sources:
            return False

        question_tokens = set(cls.tokenize(question))
        if not question_tokens:
            return False

        top = sources[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        top_tokens = set(cls.tokenize(top.get("text", "")))
        overlap = len(question_tokens & top_tokens)

        if top_score >= cls.LOCAL_MIN_BM25_SCORE:
            return True
        if overlap >= cls.LOCAL_MIN_TOKEN_OVERLAP:
            return True
        return False

    def _should_web_fallback(self, question, route, sources):
        """Only allow web fallback for questions inside the supported BI domain."""
        if route != "local":
            return False
        if self.web_rag is None:
            return False

        if not self.query_router.is_power_bi_domain(question):
            if not self._is_implicit_dax_question(question):
                return False

        return not self._local_evidence_quality(question, sources)

    @staticmethod
    def _clean_model_sources(answer):
        answer = re.sub(
            r"\n+\s*(?:Sources?|References?)\s*:.*$",
            "",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        return answer

    def _add_inline_citations(self, answer, sources):
        answer = self._clean_model_sources(answer)
        if not sources:
            return answer

        if re.search(r"\[(?:1|2|3|4|5)\]", answer):
            return answer

        blocks = re.split(r"(\n\s*\n)", answer)
        output = []

        for block in blocks:
            stripped = block.strip()
            if not stripped or stripped.startswith("#") or len(stripped) < 12:
                output.append(block)
                continue

            tokens = set(self.tokenize(stripped))
            best_number = 1
            best_overlap = -1

            for number, source in enumerate(sources, 1):
                source_tokens = set(self.tokenize(source.get("text", "")))
                overlap = len(tokens & source_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_number = number

            output.append(block.rstrip() + f" [{best_number}]")

        return "".join(output).strip()

    def _build_context(self, sources):
        parts = []
        for source in sources:
            parts.append(
                "SOURCE [{rank}]\n"
                "TITLE: {title}\n"
                "URL: {url}\n"
                "CONTENT:\n{text}".format(
                    rank=source.get("rank", 1),
                    title=source.get("title", "Source"),
                    url=source.get("url", ""),
                    text=source.get("text", ""),
                )
            )
        return "\n\n---\n\n".join(parts)

    def _generate(self, question, context, sources):
        system = (
            "You are Vinay's Power BI Copilot, a friendly and focused Business Intelligence assistant.\n"
            "Answer only questions within Power BI, Microsoft Fabric, Power Query, DAX, "
            "semantic models, Power BI Service/Desktop, BI analytics, administration, "
            "performance, or closely related BI technologies.\n"
            "Use only the supplied retrieved context for factual claims. If the context "
            "does not support the answer, say that clearly.\n"
            "For natural follow-up questions, use the current question and retrieved context "
            "to answer conversationally. Do not require the user to repeat 'Power BI' when "
            "the retrieved context clearly establishes the topic.\n"
            "Write a concise, friendly, well-structured Markdown answer. Use headings, bullets, "
            "or tables when useful.\n"
            "Do not invent URLs or sources. Do not create a Sources/References section; "
            "the application adds the source list separately.\n"
            "Place [1], [2], etc. next to factual claims when appropriate."
        )
        user = f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"

        started = time.perf_counter()
        response = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
        )
        elapsed = time.perf_counter() - started

        answer = response.choices[0].message.content or (
            "I don't have enough information in the retrieved sources."
        )
        answer = self._add_inline_citations(answer, sources)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        return answer, elapsed, input_tokens, output_tokens

    def ask(self, question):
        total_start = time.perf_counter()
        question = str(question).strip()

        route = self.query_router.classify(question)

        # The structured DAX catalog is the source of truth for implicit DAX
        # questions such as "SUM vs SUMX". This avoids requiring the user to
        # type the words "DAX" or "Power BI".
        if route == "reject" and self._is_implicit_dax_question(question):
            route = "web"

        print(f"\nQuery route: {route}")

        if route == "reject":
            answer = self.query_router.rejection_message()
            return {
                "answer": answer,
                "sources": [],
                "route": "reject",
                "retrieval_type": "out_of_domain",
                "retrieval_time": 0.0,
                "llm_time": 0.0,
                "total_time": time.perf_counter() - total_start,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        retrieval_start = time.perf_counter()

        if route == "web" and self.web_rag is not None:
            web_result = self.web_rag.search(question)
            sources = self._web_sources(web_result)
            retrieval_type = "live_web"
        else:
            if self.mode == "hybrid":
                sources = self._local_hybrid(question)
                retrieval_type = "local_hybrid"
            else:
                sources = self._local_bm25(question)
                retrieval_type = "local_bm25"

        if route == "local" and self._should_web_fallback(question, route, sources):
            print("Local evidence is weak; falling back to live web RAG.")
            web_result = self.web_rag.search(question)
            web_sources = self._web_sources(web_result)
            if web_sources:
                sources = web_sources
                retrieval_type = "local_fallback_web"
                route = "web_fallback"

        retrieval_time = time.perf_counter() - retrieval_start

        if not sources:
            answer = "I don't have enough information in the retrieved sources."
            return {
                "answer": answer,
                "sources": [],
                "route": route,
                "retrieval_type": retrieval_type,
                "retrieval_time": retrieval_time,
                "llm_time": 0.0,
                "total_time": time.perf_counter() - total_start,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        answer, llm_time, input_tokens, output_tokens = self._generate(
            question,
            self._build_context(sources),
            sources,
        )

        return {
            "answer": answer,
            "sources": sources,
            "route": route,
            "retrieval_type": retrieval_type,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": time.perf_counter() - total_start,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
