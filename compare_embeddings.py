"""
Compare embedding models on a sample corpus.

Usage:
    python scripts/compare_embeddings.py --corpus data/raw/sample.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from app.core.embeddings import compare_embedding_models


SAMPLE_TEXTS = [
    "Deep learning models can recognise emotions from facial expressions.",
    "Physiological signals like EEG and GSR are used in affective computing.",
    "Retrieval-augmented generation improves factual accuracy in LLM responses.",
    "ChromaDB stores dense vector embeddings for semantic similarity search.",
    "The transformer architecture underpins most modern NLP models.",
]

SAMPLE_QUERY = "How do neural networks process emotional signals?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=SAMPLE_QUERY)
    args = parser.parse_args()

    logger.info("Running embedding model comparison...")
    results = compare_embedding_models(
        texts=SAMPLE_TEXTS,
        query=args.query,
    )

    print(f"\nQuery: {args.query}\n")
    for model, stats in results.items():
        print(f"Model: {model}")
        print(f"  Dimensions : {stats['embedding_dim']}")
        print(f"  Encode time: {stats['encode_time_s']}s")
        print(f"  Top matches:")
        for match in stats["top_matches"]:
            print(f"    #{match['rank']} (score={match['score']}): {match['text'][:80]}")
        print()


if __name__ == "__main__":
    main()
