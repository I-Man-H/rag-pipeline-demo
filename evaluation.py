"""
Retrieval and RAG pipeline evaluation.

Implements two evaluation layers:

1. Retrieval metrics (no LLM required):
   - Hit Rate    — did the correct chunk appear in top-k?
   - MRR         — mean reciprocal rank of the first correct chunk
   - Precision@k — fraction of retrieved chunks that are relevant

2. RAG quality metrics (requires LLM):
   - Faithfulness   — is the answer grounded in the retrieved context?
   - Answer relevancy — does the answer address the question?
   - Context recall  — does the context contain the ground truth?

   These use the RAGAS framework (https://docs.ragas.io).
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Retrieval-level evaluation (no LLM)
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    queries: List[str],
    ground_truth_docs: List[List[str]],
    retrieved_docs: List[List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate retrieval quality against ground truth document IDs.

    Args:
        queries:            List of query strings.
        ground_truth_docs:  For each query, list of relevant doc source names.
        retrieved_docs:     For each query, list of retrieved doc source names.
        k:                  Cutoff for Precision@k.

    Returns:
        Dictionary with hit_rate, mrr, and precision_at_k.
    """
    hit_rates, reciprocal_ranks, precisions = [], [], []

    for gt_docs, ret_docs in zip(ground_truth_docs, retrieved_docs):
        gt_set = set(gt_docs)
        ret_at_k = ret_docs[:k]

        # Hit Rate
        hit = any(doc in gt_set for doc in ret_at_k)
        hit_rates.append(float(hit))

        # MRR
        rr = 0.0
        for rank, doc in enumerate(ret_at_k, start=1):
            if doc in gt_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Precision@k
        relevant = sum(1 for doc in ret_at_k if doc in gt_set)
        precisions.append(relevant / k)

    metrics = {
        "hit_rate": round(np.mean(hit_rates), 4),
        "mrr": round(np.mean(reciprocal_ranks), 4),
        f"precision_at_{k}": round(np.mean(precisions), 4),
        "num_queries": len(queries),
    }

    logger.info(f"Retrieval evaluation: {metrics}")
    return metrics


def build_retrieval_report(
    queries: List[str],
    ground_truth_docs: List[List[str]],
    retrieved_docs: List[List[str]],
    k: int = 5,
) -> pd.DataFrame:
    """Return a per-query breakdown as a DataFrame for inspection."""
    rows = []
    for query, gt_docs, ret_docs in zip(
        queries, ground_truth_docs, retrieved_docs
    ):
        gt_set = set(gt_docs)
        ret_at_k = ret_docs[:k]
        hit = any(doc in gt_set for doc in ret_at_k)
        rr = next(
            (1.0 / (r + 1) for r, d in enumerate(ret_at_k) if d in gt_set),
            0.0,
        )
        rows.append({
            "query": query,
            "hit": hit,
            "reciprocal_rank": round(rr, 4),
            "retrieved": ret_at_k,
            "ground_truth": gt_docs,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RAG-level evaluation with RAGAS
# ---------------------------------------------------------------------------

def evaluate_rag_pipeline(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate end-to-end RAG quality using the RAGAS framework.

    Metrics computed:
      - faithfulness      (answer grounded in context)
      - answer_relevancy  (answer relevant to question)
      - context_recall    (context covers ground truth) — requires ground_truths

    Args:
        questions:     List of user questions.
        answers:       Generated answers from the RAG chain.
        contexts:      Retrieved context chunks per question.
        ground_truths: Optional reference answers for context recall.

    Returns:
        Dictionary of RAGAS metric scores.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset
    except ImportError:
        logger.error(
            "RAGAS not installed. Run: pip install ragas datasets"
        )
        return {}

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    scores = dict(result)
    logger.info(f"RAGAS evaluation: {scores}")
    return scores
