import json
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    BM25_TOP_K,
    EMBEDDING_TOP_K,
    RRF_TOP_K,
    FINAL_TOP_K,
    RRF_K,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "retrieval_diagnostic_results.json"


# ============================================================
# EVALUATION QUESTIONS
#
# expected_document = document that should be the primary
# source for answering the question.
# ============================================================

QUESTIONS = [

    # --------------------------------------------------------
    # DIRECTQUERY
    # --------------------------------------------------------

    {
        "question":
            "What is DirectQuery in Power BI?",

        "expected_document":
            "pbi_web_001"
    },

    {
        "question":
            "How does DirectQuery work in Power BI?",

        "expected_document":
            "pbi_web_002"
    },

    {
        "question":
            "What are the limitations of DirectQuery?",

        "expected_document":
            "pbi_web_002"
    },


    # --------------------------------------------------------
    # INCREMENTAL REFRESH
    # --------------------------------------------------------

    {
        "question":
            "What is incremental refresh in Power BI?",

        "expected_document":
            "pbi_web_003"
    },

    {
        "question":
            "How do I configure incremental refresh?",

        "expected_document":
            "pbi_web_003"
    },

    {
        "question":
            "How does incremental refresh improve refresh performance?",

        "expected_document":
            "pbi_web_003"
    },


    # --------------------------------------------------------
    # QUERY FOLDING
    # --------------------------------------------------------

    {
        "question":
            "What is query folding in Power BI?",

        "expected_document":
            "pbi_web_004"
    },

    {
        "question":
            "Why is query folding important?",

        "expected_document":
            "pbi_web_004"
    },


    # --------------------------------------------------------
    # STORAGE MODES
    # --------------------------------------------------------

    {
        "question":
            "What are the storage modes in Power BI?",

        "expected_document":
            "pbi_web_005"
    },

    {
        "question":
            "What is Import mode in Power BI?",

        "expected_document":
            "pbi_web_005"
    },

    {
        "question":
            "What is DirectQuery storage mode?",

        "expected_document":
            "pbi_web_005"
    },


    # --------------------------------------------------------
    # COMPOSITE MODELS
    # --------------------------------------------------------

    {
        "question":
            "What are composite models in Power BI?",

        "expected_document":
            "pbi_web_006"
    },

    {
        "question":
            "When should I use a composite model?",

        "expected_document":
            "pbi_web_006"
    },


    # --------------------------------------------------------
    # SEMANTIC MODEL MODES
    # --------------------------------------------------------

    {
        "question":
            "What are semantic model modes in Power BI?",

        "expected_document":
            "pbi_web_007"
    },


    # --------------------------------------------------------
    # DATA REFRESH
    # --------------------------------------------------------

    {
        "question":
            "How does data refresh work in Power BI?",

        "expected_document":
            "pbi_web_008"
    },

    {
        "question":
            "How can I refresh data in Power BI?",

        "expected_document":
            "pbi_web_008"
    },


    # --------------------------------------------------------
    # AUTOMATIC PAGE REFRESH
    # --------------------------------------------------------

    {
        "question":
            "What is automatic page refresh in Power BI?",

        "expected_document":
            "pbi_web_009"
    },


    # --------------------------------------------------------
    # STAR SCHEMA
    # --------------------------------------------------------

    {
        "question":
            "What is a star schema in Power BI?",

        "expected_document":
            "pbi_web_011"
    },

    {
        "question":
            "Why is star schema important for Power BI?",

        "expected_document":
            "pbi_web_011"
    },


    # --------------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------------

    {
        "question":
            "How do relationships work in Power BI?",

        "expected_document":
            "pbi_web_012"
    },

    {
        "question":
            "What are many-to-many relationships in Power BI?",

        "expected_document":
            "pbi_web_013"
    },

    {
        "question":
            "What is the difference between active and inactive relationships?",

        "expected_document":
            "pbi_web_014"
    },


    # --------------------------------------------------------
    # DATA REDUCTION
    # --------------------------------------------------------

    {
        "question":
            "How can I reduce data in an Import model?",

        "expected_document":
            "pbi_web_015"
    },


    # --------------------------------------------------------
    # CALCULATIONS
    # --------------------------------------------------------

    {
        "question":
            "What calculation options are available in Power BI?",

        "expected_document":
            "pbi_web_016"
    },


    # --------------------------------------------------------
    # GATEWAY
    # --------------------------------------------------------

    {
        "question":
            "What is an on-premises data gateway?",

        "expected_document":
            "pbi_web_017"
    },

    {
        "question":
            "How does the on-premises data gateway architecture work?",

        "expected_document":
            "pbi_web_018"
    },

    {
        "question":
            "How do I install an on-premises data gateway?",

        "expected_document":
            "pbi_web_019"
    },

    {
        "question":
            "How do I troubleshoot an on-premises data gateway?",

        "expected_document":
            "pbi_web_020"
    },


    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    {
        "question":
            "What are Power BI reports?",

        "expected_document":
            "pbi_web_024"
    },


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    {
        "question":
            "How can I optimize a Power BI model for performance?",

        "expected_document":
            "pbi_web_026"
    },


    # --------------------------------------------------------
    # AUDITING
    # --------------------------------------------------------

    {
        "question":
            "What is data-level auditing in Power BI?",

        "expected_document":
            "pbi_web_027"
    },

    {
        "question":
            "How can Power BI implementation planning help with auditing?",

        "expected_document":
            "pbi_web_027"
    },

]


# ============================================================
# HELPERS
# ============================================================

def tokenize(text):

    return re.findall(
        r"\w+",
        str(text).lower()
    )


def normalize(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("POWER BI RETRIEVAL DIAGNOSTIC")
print("=" * 100)

print("\nLoading chunks...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    raw_chunks = json.load(file)


# Remove duplicate chunk IDs
unique_chunks = {}

for chunk in raw_chunks:

    unique_chunks[
        chunk["chunk_id"]
    ] = chunk


chunks = list(
    unique_chunks.values()
)


print(
    f"Chunks loaded       : {len(chunks)}"
)

print(
    f"Evaluation questions: {len(QUESTIONS)}"
)


documents = [
    normalize(chunk["text"])
    for chunk in chunks
]


# ============================================================
# BUILD BM25
# ============================================================

print("\nPreparing BM25...")

bm25 = BM25Okapi([

    tokenize(text)

    for text in documents

])


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print("\nCreating document embeddings...")

embedding_start = time.perf_counter()

document_embeddings = embedding_model.encode(

    documents,

    batch_size=32,

    show_progress_bar=True,

    normalize_embeddings=True

)


embedding_time = (
    time.perf_counter()
    -
    embedding_start
)


print(
    f"Embedding creation time: "
    f"{embedding_time:.3f}s"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    bm25_indices,
    embedding_indices
):

    scores = {}


    for rank, index in enumerate(
        bm25_indices,
        start=1
    ):

        scores[index] = (

            scores.get(index, 0.0)

            +

            1.0 /
            (RRF_K + rank)

        )


    for rank, index in enumerate(
        embedding_indices,
        start=1
    ):

        scores[index] = (

            scores.get(index, 0.0)

            +

            1.0 /
            (RRF_K + rank)

        )


    ranked = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )


    return [

        index

        for index, score

        in ranked[:RRF_TOP_K]

    ]


# ============================================================
# RANK HELPERS
# ============================================================

def get_document_rank(
    indices,
    expected_document_id
):

    for rank, index in enumerate(
        indices,
        start=1
    ):

        if (
            chunks[index]["document_id"]
            ==
            expected_document_id
        ):

            return rank


    return None


def format_results(
    indices,
    expected_document_id
):

    results = []


    for rank, index in enumerate(
        indices,
        start=1
    ):

        results.append({

            "rank":
                rank,

            "document_id":
                chunks[index]["document_id"],

            "title":
                chunks[index]["title"],

            "chunk_id":
                chunks[index]["chunk_id"],

            "expected":
                (
                    chunks[index]["document_id"]
                    ==
                    expected_document_id
                )

        })


    return results


# ============================================================
# EVALUATION
# ============================================================

results = []


bm25_top1 = 0
bm25_top3 = 0
bm25_top5 = 0
bm25_top10 = 0


embedding_top1 = 0
embedding_top3 = 0
embedding_top5 = 0
embedding_top10 = 0


rrf_top1 = 0
rrf_top3 = 0
rrf_top5 = 0
rrf_top10 = 0


rerank_top1 = 0
rerank_top3 = 0
rerank_top5 = 0
rerank_top10 = 0


reranker_improved = 0
reranker_degraded = 0
reranker_unchanged = 0


print("\n")
print("=" * 100)
print("RUNNING RETRIEVAL TESTS")
print("=" * 100)


for number, item in enumerate(
    QUESTIONS,
    start=1
):

    question = item["question"]

    expected_document = (
        item["expected_document"]
    )


    print("\n" + "-" * 100)

    print(
        f"[{number}/{len(QUESTIONS)}]"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Expected: {expected_document}"
    )


    # ========================================================
    # BM25
    # ========================================================

    bm25_scores = bm25.get_scores(
        tokenize(question)
    )


    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:BM25_TOP_K]


    bm25_rank = get_document_rank(
        bm25_indices,
        expected_document
    )


    # ========================================================
    # EMBEDDING
    # ========================================================

    query_embedding = (
        embedding_model.encode(
            question,
            normalize_embeddings=True
        )
    )


    embedding_scores = (
        document_embeddings
        @
        query_embedding
    )


    embedding_indices = np.argsort(
        embedding_scores
    )[::-1][:EMBEDDING_TOP_K]


    embedding_rank = get_document_rank(
        embedding_indices,
        expected_document
    )


    # ========================================================
    # RRF
    # ========================================================

    rrf_indices = (
        reciprocal_rank_fusion(
            bm25_indices,
            embedding_indices
        )
    )


    rrf_rank = get_document_rank(
        rrf_indices,
        expected_document
    )


    # ========================================================
    # RERANKER
    # ========================================================

    rerank_pairs = [

        [
            question,
            documents[index]
        ]

        for index in rrf_indices

    ]


    rerank_scores = (
        reranker.predict(
            rerank_pairs
        )
    )


    reranked = sorted(

        zip(
            rrf_indices,
            rerank_scores
        ),

        key=lambda x: x[1],

        reverse=True

    )


    reranked_indices = [

        index

        for index, score

        in reranked

    ]


    rerank_rank = get_document_rank(
        reranked_indices,
        expected_document
    )


    # ========================================================
    # METRICS
    # ========================================================

    if bm25_rank == 1:
        bm25_top1 += 1

    if bm25_rank is not None and bm25_rank <= 3:
        bm25_top3 += 1

    if bm25_rank is not None and bm25_rank <= 5:
        bm25_top5 += 1

    if bm25_rank is not None and bm25_rank <= 10:
        bm25_top10 += 1


    if embedding_rank == 1:
        embedding_top1 += 1

    if embedding_rank is not None and embedding_rank <= 3:
        embedding_top3 += 1

    if embedding_rank is not None and embedding_rank <= 5:
        embedding_top5 += 1

    if embedding_rank is not None and embedding_rank <= 10:
        embedding_top10 += 1


    if rrf_rank == 1:
        rrf_top1 += 1

    if rrf_rank is not None and rrf_rank <= 3:
        rrf_top3 += 1

    if rrf_rank is not None and rrf_rank <= 5:
        rrf_top5 += 1

    if rrf_rank is not None and rrf_rank <= 10:
        rrf_top10 += 1


    if rerank_rank == 1:
        rerank_top1 += 1

    if rerank_rank is not None and rerank_rank <= 3:
        rerank_top3 += 1

    if rerank_rank is not None and rerank_rank <= 5:
        rerank_top5 += 1

    if rerank_rank is not None and rerank_rank <= 10:
        rerank_top10 += 1


    # ========================================================
    # RERANKER EFFECT
    # ========================================================

    if (
        rrf_rank is not None
        and
        rerank_rank is not None
    ):

        if rerank_rank < rrf_rank:

            reranker_improved += 1

        elif rerank_rank > rrf_rank:

            reranker_degraded += 1

        else:

            reranker_unchanged += 1


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"BM25       rank: {bm25_rank}"
    )

    print(
        f"Embedding  rank: {embedding_rank}"
    )

    print(
        f"RRF        rank: {rrf_rank}"
    )

    print(
        f"Reranker   rank: {rerank_rank}"
    )


    # --------------------------------------------------------
    # Show final top 3
    # --------------------------------------------------------

    print("\nFinal Reranked Top 3:")


    for rank, index in enumerate(
        reranked_indices[:FINAL_TOP_K],
        start=1
    ):

        marker = (

            " <-- EXPECTED"

            if chunks[index]["document_id"]
            ==
            expected_document

            else ""

        )


        print(

            f"  {rank}. "

            f"{chunks[index]['document_id']} "

            f"| "

            f"{chunks[index]['title']}"

            f"{marker}"

        )


    results.append({

        "question":
            question,

        "expected_document":
            expected_document,

        "bm25_rank":
            bm25_rank,

        "embedding_rank":
            embedding_rank,

        "rrf_rank":
            rrf_rank,

        "reranker_rank":
            rerank_rank,

        "bm25_results":
            format_results(
                bm25_indices,
                expected_document
            ),

        "embedding_results":
            format_results(
                embedding_indices,
                expected_document
            ),

        "rrf_results":
            format_results(
                rrf_indices,
                expected_document
            ),

        "reranked_results":
            format_results(
                reranked_indices,
                expected_document
            )

    })


# ============================================================
# FINAL SUMMARY
# ============================================================

total = len(QUESTIONS)


def percentage(value):

    return (
        value / total * 100
    )


print("\n")
print("=" * 100)
print("FINAL RETRIEVAL DIAGNOSTIC")
print("=" * 100)


print("\nBM25")

print(
    f"Top 1 : {bm25_top1}/{total} "
    f"({percentage(bm25_top1):.2f}%)"
)

print(
    f"Top 3 : {bm25_top3}/{total} "
    f"({percentage(bm25_top3):.2f}%)"
)

print(
    f"Top 5 : {bm25_top5}/{total} "
    f"({percentage(bm25_top5):.2f}%)"
)

print(
    f"Top 10: {bm25_top10}/{total} "
    f"({percentage(bm25_top10):.2f}%)"
)


print("\nEmbedding")

print(
    f"Top 1 : {embedding_top1}/{total} "
    f"({percentage(embedding_top1):.2f}%)"
)

print(
    f"Top 3 : {embedding_top3}/{total} "
    f"({percentage(embedding_top3):.2f}%)"
)

print(
    f"Top 5 : {embedding_top5}/{total} "
    f"({percentage(embedding_top5):.2f}%)"
)

print(
    f"Top 10: {embedding_top10}/{total} "
    f"({percentage(embedding_top10):.2f}%)"
)


print("\nRRF")

print(
    f"Top 1 : {rrf_top1}/{total} "
    f"({percentage(rrf_top1):.2f}%)"
)

print(
    f"Top 3 : {rrf_top3}/{total} "
    f"({percentage(rrf_top3):.2f}%)"
)

print(
    f"Top 5 : {rrf_top5}/{total} "
    f"({percentage(rrf_top5):.2f}%)"
)

print(
    f"Top 10: {rrf_top10}/{total} "
    f"({percentage(rrf_top10):.2f}%)"
)


print("\nReranker")

print(
    f"Top 1 : {rerank_top1}/{total} "
    f"({percentage(rerank_top1):.2f}%)"
)

print(
    f"Top 3 : {rerank_top3}/{total} "
    f"({percentage(rerank_top3):.2f}%)"
)

print(
    f"Top 5 : {rerank_top5}/{total} "
    f"({percentage(rerank_top5):.2f}%)"
)

print(
    f"Top 10: {rerank_top10}/{total} "
    f"({percentage(rerank_top10):.2f}%)"
)


print("\nReranker Effect")

print(
    f"Improved : {reranker_improved}"
)

print(
    f"Degraded : {reranker_degraded}"
)

print(
    f"Unchanged: {reranker_unchanged}"
)


print("\nEmbedding creation time:")

print(
    f"{embedding_time:.3f}s"
)


# ============================================================
# SAVE
# ============================================================

output = {

    "total_questions":
        total,

    "embedding_creation_time":
        embedding_time,

    "bm25": {

        "top1":
            bm25_top1,

        "top3":
            bm25_top3,

        "top5":
            bm25_top5,

        "top10":
            bm25_top10

    },

    "embedding": {

        "top1":
            embedding_top1,

        "top3":
            embedding_top3,

        "top5":
            embedding_top5,

        "top10":
            embedding_top10

    },

    "rrf": {

        "top1":
            rrf_top1,

        "top3":
            rrf_top3,

        "top5":
            rrf_top5,

        "top10":
            rrf_top10

    },

    "reranker": {

        "top1":
            rerank_top1,

        "top3":
            rerank_top3,

        "top5":
            rerank_top5,

        "top10":
            rerank_top10,

        "improved":
            reranker_improved,

        "degraded":
            reranker_degraded,

        "unchanged":
            reranker_unchanged

    },

    "questions":
        results

}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print(
    f"Results saved to: {OUTPUT_FILE}"
)

print("=" * 100)
