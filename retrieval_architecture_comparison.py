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
    RRF_K,
)


# ============================================================
# RETRIEVAL ARCHITECTURE COMPARISON
# ============================================================
#
# Tests four architectures in ONE execution:
#
# A. RRF baseline
#
# B. RRF + reranker
#
# C. Weighted RRF + reranker
#
# D. Weighted RRF + reranker + document diversity
#
# IMPORTANT:
# No LLM calls are made.
#
# This test evaluates retrieval only.
# ============================================================


OUTPUT_FILE = "retrieval_architecture_comparison.json"


# ============================================================
# TEST QUESTIONS
# ============================================================

QUESTIONS = [

    {
        "question": "What is DirectQuery in Power BI?",
        "expected_document": "pbi_web_001"
    },

    {
        "question": "How does DirectQuery work in Power BI?",
        "expected_document": "pbi_web_002"
    },

    {
        "question": "What are the limitations of DirectQuery?",
        "expected_document": "pbi_web_002"
    },

    {
        "question": "What is incremental refresh in Power BI?",
        "expected_document": "pbi_web_003"
    },

    {
        "question": "How do I configure incremental refresh?",
        "expected_document": "pbi_web_003"
    },

    {
        "question": "How does incremental refresh improve refresh performance?",
        "expected_document": "pbi_web_003"
    },

    {
        "question": "What is query folding in Power BI?",
        "expected_document": "pbi_web_004"
    },

    {
        "question": "Why is query folding important?",
        "expected_document": "pbi_web_004"
    },

    {
        "question": "What are the storage modes in Power BI?",
        "expected_document": "pbi_web_005"
    },

    {
        "question": "What is Import mode in Power BI?",
        "expected_document": "pbi_web_005"
    },

    {
        "question": "What is DirectQuery storage mode?",
        "expected_document": "pbi_web_005"
    },

    {
        "question": "What are composite models in Power BI?",
        "expected_document": "pbi_web_006"
    },

    {
        "question": "When should I use a composite model?",
        "expected_document": "pbi_web_006"
    },

    {
        "question": "What are semantic model modes in Power BI?",
        "expected_document": "pbi_web_007"
    },

    {
        "question": "How does data refresh work in Power BI?",
        "expected_document": "pbi_web_008"
    },

    {
        "question": "How can I refresh data in Power BI?",
        "expected_document": "pbi_web_008"
    },

    {
        "question": "What is automatic page refresh in Power BI?",
        "expected_document": "pbi_web_009"
    },

    {
        "question": "What is a star schema in Power BI?",
        "expected_document": "pbi_web_011"
    },

    {
        "question": "Why is star schema important for Power BI?",
        "expected_document": "pbi_web_011"
    },

    {
        "question": "How do relationships work in Power BI?",
        "expected_document": "pbi_web_012"
    },

    {
        "question": "What are many-to-many relationships in Power BI?",
        "expected_document": "pbi_web_013"
    },

    {
        "question": "What is the difference between active and inactive relationships?",
        "expected_document": "pbi_web_014"
    },

    {
        "question": "How can I reduce data in an Import model?",
        "expected_document": "pbi_web_015"
    },

    {
        "question": "What calculation options are available in Power BI?",
        "expected_document": "pbi_web_016"
    },

    {
        "question": "What is an on-premises data gateway?",
        "expected_document": "pbi_web_017"
    },

    {
        "question": "How does the on-premises data gateway architecture work?",
        "expected_document": "pbi_web_018"
    },

    {
        "question": "How do I install an on-premises data gateway?",
        "expected_document": "pbi_web_019"
    },

    {
        "question": "How do I troubleshoot an on-premises data gateway?",
        "expected_document": "pbi_web_020"
    },

    {
        "question": "What are Power BI reports?",
        "expected_document": "pbi_web_024"
    },

    {
        "question": "How can I optimize a Power BI model for performance?",
        "expected_document": "pbi_web_026"
    },

    {
        "question": "What is data-level auditing in Power BI?",
        "expected_document": "pbi_web_027"
    },

    {
        "question": "How can Power BI implementation planning help with auditing?",
        "expected_document": "pbi_web_027"
    },

]


# ============================================================
# CONFIGURATION
# ============================================================

BM25_TOP_K = 20
EMBEDDING_TOP_K = 20

RRF_TOP_K = 10

RERANK_TOP_K = 10

FINAL_TOP_K = 3

# How much the RRF score contributes to the hybrid score.
#
# 0.70 means:
#
#     70% RRF
#     30% reranker
#
RRF_WEIGHT = 0.70
RERANK_WEIGHT = 0.30


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


def rank_of_document(
    indices,
    chunks,
    expected_document
):

    for rank, index in enumerate(
        indices,
        start=1
    ):

        if (
            chunks[index]["document_id"]
            ==
            expected_document
        ):

            return rank

    return None


def unique_document_indices(
    indices,
    chunks,
    limit
):

    selected = []

    seen_documents = set()

    for index in indices:

        document_id = (
            chunks[index]["document_id"]
        )

        if document_id in seen_documents:
            continue

        seen_documents.add(
            document_id
        )

        selected.append(
            index
        )

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# RRF
# ============================================================

def calculate_rrf(
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


    return ranked


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:

        return []

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.ones(
            len(scores)
        )

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 110)
print("POWER BI RETRIEVAL ARCHITECTURE COMPARISON")
print("=" * 110)

print()
print("This test compares:")
print()
print("A. RRF baseline")
print("B. RRF + reranker")
print("C. Weighted RRF + reranker")
print("D. Weighted RRF + reranker + document diversity")
print()
print("LLM calls: 0")
print()


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 110)
print("LOADING CHUNKS")
print("=" * 110)

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
    f"Unique documents    : "
    f"{len(set(x['document_id'] for x in chunks))}"
)

print(
    f"Evaluation questions: "
    f"{len(QUESTIONS)}"
)


documents = [

    normalize(
        chunk["text"]
    )

    for chunk in chunks

]


# ============================================================
# BM25
# ============================================================

print()
print("=" * 110)
print("PREPARING BM25")
print("=" * 110)

bm25 = BM25Okapi(

    [
        tokenize(text)
        for text in documents
    ]

)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print()
print("=" * 110)
print("LOADING EMBEDDING MODEL")
print("=" * 110)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print()
print("Creating document embeddings...")

embedding_start = time.perf_counter()

document_embeddings = (
    embedding_model.encode(

        documents,

        batch_size=32,

        show_progress_bar=True,

        normalize_embeddings=True

    )
)

embedding_creation_time = (
    time.perf_counter()
    -
    embedding_start
)


print(
    f"Embedding creation time: "
    f"{embedding_creation_time:.3f}s"
)


# ============================================================
# RERANKER
# ============================================================

print()
print("=" * 110)
print("LOADING RERANKER")
print("=" * 110)

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ============================================================
# METRICS
# ============================================================

architectures = {

    "A_RRF": {

        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,

    },

    "B_RRF_RERANKER": {

        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,

    },

    "C_WEIGHTED_RRF_RERANKER": {

        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,

    },

    "D_WEIGHTED_RRF_RERANKER_DIVERSITY": {

        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,

    }

}


reranker_effect = {

    "improved": 0,

    "degraded": 0,

    "unchanged": 0

}


question_results = []


# ============================================================
# EVALUATE QUESTIONS
# ============================================================

print()
print("=" * 110)
print("RUNNING ARCHITECTURE COMPARISON")
print("=" * 110)


for number, item in enumerate(
    QUESTIONS,
    start=1
):

    question = item["question"]

    expected_document = (
        item["expected_document"]
    )


    print()
    print("-" * 110)

    print(
        f"[{number}/{len(QUESTIONS)}] "
        f"{question}"
    )

    print(
        f"Expected document: "
        f"{expected_document}"
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


    # ========================================================
    # RRF
    # ========================================================

    rrf_ranked = calculate_rrf(

        bm25_indices,

        embedding_indices

    )


    rrf_indices = [

        index

        for index, score
        in rrf_ranked[:RRF_TOP_K]

    ]


    rrf_score_map = {

        index: score

        for index, score
        in rrf_ranked

    }


    # ========================================================
    # RRF BASELINE
    # ========================================================

    architecture_a = rrf_indices


    # ========================================================
    # RERANKER
    # ========================================================

    rerank_candidates = rrf_indices[
        :RERANK_TOP_K
    ]


    rerank_pairs = [

        [
            question,
            documents[index]
        ]

        for index in rerank_candidates

    ]


    rerank_scores = (
        reranker.predict(
            rerank_pairs
        )
    )


    reranked_pairs = sorted(

        zip(
            rerank_candidates,
            rerank_scores
        ),

        key=lambda x: x[1],

        reverse=True

    )


    architecture_b = [

        index

        for index, score
        in reranked_pairs

    ]


    # ========================================================
    # RERANKER EFFECT
    # ========================================================

    rrf_expected_rank = (
        rank_of_document(
            architecture_a,
            chunks,
            expected_document
        )
    )


    reranker_expected_rank = (
        rank_of_document(
            architecture_b,
            chunks,
            expected_document
        )
    )


    if (

        rrf_expected_rank is not None

        and

        reranker_expected_rank is not None

    ):

        if (
            reranker_expected_rank
            <
            rrf_expected_rank
        ):

            reranker_effect["improved"] += 1

        elif (
            reranker_expected_rank
            >
            rrf_expected_rank
        ):

            reranker_effect["degraded"] += 1

        else:

            reranker_effect["unchanged"] += 1


    # ========================================================
    # WEIGHTED RRF + RERANKER
    # ========================================================

    candidate_indices = [

        index

        for index, score
        in rrf_ranked[:RERANK_TOP_K]

    ]


    candidate_rrf_scores = [

        rrf_score_map[index]

        for index in candidate_indices

    ]


    candidate_rerank_scores = [

        score

        for index, score
        in reranked_pairs

    ]


    # Reranker scores need to be mapped by index.

    rerank_score_map = {

        index: float(score)

        for index, score
        in reranked_pairs

    }


    rrf_normalized = (
        normalize_scores(
            candidate_rrf_scores
        )
    )


    rerank_normalized = (
        normalize_scores([

            rerank_score_map[index]

            for index in candidate_indices

        ])
    )


    weighted_scores = {}


    for position, index in enumerate(
        candidate_indices
    ):

        weighted_scores[index] = (

            RRF_WEIGHT
            *
            float(
                rrf_normalized[position]
            )

            +

            RERANK_WEIGHT
            *
            float(
                rerank_normalized[position]
            )

        )


    weighted_ranked = sorted(

        weighted_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )


    architecture_c = [

        index

        for index, score
        in weighted_ranked

    ]


    # ========================================================
    # DOCUMENT DIVERSITY
    # ========================================================
    #
    # We first create a larger weighted candidate list.
    #
    # Then select the best chunk from each unique document.
    #
    # This prevents:
    #
    # pbi_web_003 chunk 1
    # pbi_web_003 chunk 2
    # pbi_web_003 chunk 3
    #
    # from consuming the entire final context.
    #
    # ========================================================

    architecture_d = unique_document_indices(

        architecture_c,

        chunks,

        FINAL_TOP_K

    )


    # ========================================================
    # TOP-K METRICS
    # ========================================================

    architecture_outputs = {

        "A_RRF":
            architecture_a,

        "B_RRF_RERANKER":
            architecture_b,

        "C_WEIGHTED_RRF_RERANKER":
            architecture_c,

        "D_WEIGHTED_RRF_RERANKER_DIVERSITY":
            architecture_d

    }


    per_question = {

        "question":
            question,

        "expected_document":
            expected_document,

        "architectures": {}

    }


    for architecture_name, indices in (
        architecture_outputs.items()
    ):

        rank = rank_of_document(

            indices,

            chunks,

            expected_document

        )


        per_question[
            "architectures"
        ][
            architecture_name
        ] = {

            "rank":
                rank,

            "top_results": [

                {

                    "rank":
                        position + 1,

                    "document_id":
                        chunks[index][
                            "document_id"
                        ],

                    "title":
                        chunks[index][
                            "title"
                        ],

                    "chunk_id":
                        chunks[index][
                            "chunk_id"
                        ],

                    "expected":
                        (
                            chunks[index][
                                "document_id"
                            ]
                            ==
                            expected_document
                        )

                }

                for position, index
                in enumerate(
                    indices[:10]
                )

            ]

        }


        if rank == 1:

            architectures[
                architecture_name
            ]["top1"] += 1


        if (
            rank is not None
            and
            rank <= 3
        ):

            architectures[
                architecture_name
            ]["top3"] += 1


        if (
            rank is not None
            and
            rank <= 5
        ):

            architectures[
                architecture_name
            ]["top5"] += 1


        if (
            rank is not None
            and
            rank <= 10
        ):

            architectures[
                architecture_name
            ]["top10"] += 1


    question_results.append(
        per_question
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    for architecture_name in (

        "A_RRF",

        "B_RRF_RERANKER",

        "C_WEIGHTED_RRF_RERANKER",

        "D_WEIGHTED_RRF_RERANKER_DIVERSITY"

    ):

        rank = (
            per_question[
                "architectures"
            ][
                architecture_name
            ][
                "rank"
            ]
        )


        print(
            f"{architecture_name:40s}: "
            f"rank={rank}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

total = len(
    QUESTIONS
)


def pct(value):

    return (
        value
        /
        total
        *
        100
    )


print()
print()
print("=" * 110)
print("FINAL ARCHITECTURE COMPARISON")
print("=" * 110)


print()

print(
    f"{'Architecture':45s}"
    f"{'Top-1':>12s}"
    f"{'Top-3':>12s}"
    f"{'Top-5':>12s}"
    f"{'Top-10':>12s}"
)

print("-" * 110)


for architecture_name, metrics in (
    architectures.items()
):

    print(

        f"{architecture_name:45s}"

        f"{metrics['top1']:>5}/{total}"
        f" ({pct(metrics['top1']):5.2f}%)"

        f"{metrics['top3']:>5}/{total}"
        f" ({pct(metrics['top3']):5.2f}%)"

        f"{metrics['top5']:>5}/{total}"
        f" ({pct(metrics['top5']):5.2f}%)"

        f"{metrics['top10']:>5}/{total}"
        f" ({pct(metrics['top10']):5.2f}%)"

    )


# ============================================================
# RERANKER EFFECT
# ============================================================

print()
print("=" * 110)
print("CURRENT RERANKER EFFECT")
print("=" * 110)

print(
    f"Improved : "
    f"{reranker_effect['improved']}"
)

print(
    f"Degraded : "
    f"{reranker_effect['degraded']}"
)

print(
    f"Unchanged: "
    f"{reranker_effect['unchanged']}"
)


# ============================================================
# BEST ARCHITECTURE
# ============================================================

best_architecture = max(

    architectures.items(),

    key=lambda item: (

        item[1]["top1"],

        item[1]["top3"],

        item[1]["top5"],

        item[1]["top10"]

    )

)


print()
print("=" * 110)
print("CURRENT WINNER")
print("=" * 110)

print(
    best_architecture[0]
)

print(
    f"Top-1 : "
    f"{best_architecture[1]['top1']}/{total} "
    f"({pct(best_architecture[1]['top1']):.2f}%)"
)

print(
    f"Top-3 : "
    f"{best_architecture[1]['top3']}/{total} "
    f"({pct(best_architecture[1]['top3']):.2f}%)"
)

print(
    f"Top-5 : "
    f"{best_architecture[1]['top5']}/{total} "
    f"({pct(best_architecture[1]['top5']):.2f}%)"
)

print(
    f"Top-10: "
    f"{best_architecture[1]['top10']}/{total} "
    f"({pct(best_architecture[1]['top10']):.2f}%)"
)


# ============================================================
# QUESTIONS WHERE ARCHITECTURES DISAGREE
# ============================================================

print()
print("=" * 110)
print("IMPORTANT DISAGREEMENTS")
print("=" * 110)


for item in question_results:

    ranks = {}

    for architecture_name, data in (
        item["architectures"].items()
    ):

        ranks[
            architecture_name
        ] = data["rank"]


    unique_ranks = set(
        ranks.values()
    )


    if len(unique_ranks) > 1:

        print()
        print(
            "Question:",
            item["question"]
        )

        print(
            "Expected:",
            item["expected_document"]
        )


        for name, rank in ranks.items():

            print(
                f"  {name:40s}: "
                f"{rank}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "configuration": {

        "bm25_top_k":
            BM25_TOP_K,

        "embedding_top_k":
            EMBEDDING_TOP_K,

        "rrf_top_k":
            RRF_TOP_K,

        "rerank_top_k":
            RERANK_TOP_K,

        "final_top_k":
            FINAL_TOP_K,

        "rrf_weight":
            RRF_WEIGHT,

        "rerank_weight":
            RERANK_WEIGHT

    },

    "total_questions":
        total,

    "embedding_creation_time":
        embedding_creation_time,

    "architectures":
        architectures,

    "reranker_effect":
        reranker_effect,

    "best_architecture":
        best_architecture[0],

    "questions":
        question_results

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


print()
print("=" * 110)

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print("=" * 110)

