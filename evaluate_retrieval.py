"""
Evaluate retrieval quality against a labelled test set.

Usage:
    python scripts/evaluate_retrieval.py --test-file data/eval_set.json --k 5

Expected JSON format:
[
  {
    "query": "What is the main conclusion?",
    "relevant_sources": ["report_2024.pdf"]
  }
]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from app.core.retrieval import get_vector_store, similarity_search
from app.core.evaluation import evaluate_retrieval, build_retrieval_report
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to JSON evaluation set.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to retrieve per query.")
    parser.add_argument("--collection", type=str,
                        default=settings.chroma_collection_name)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV path to save per-query report.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.test_file) as f:
        eval_set = json.load(f)

    store = get_vector_store(collection_name=args.collection)

    queries, ground_truths, retrieved = [], [], []

    for item in eval_set:
        query = item["query"]
        gt = item["relevant_sources"]

        results = similarity_search(store, query, k=args.k)
        ret_sources = [doc.metadata.get("source", "") for doc in results]

        queries.append(query)
        ground_truths.append(gt)
        retrieved.append(ret_sources)

    metrics = evaluate_retrieval(queries, ground_truths, retrieved, k=args.k)
    logger.info(f"\nEvaluation Results (k={args.k}):")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

    if args.output:
        report = build_retrieval_report(queries, ground_truths, retrieved, k=args.k)
        report.to_csv(args.output, index=False)
        logger.info(f"Per-query report saved to: {args.output}")


if __name__ == "__main__":
    main()
