#!/usr/bin/env python3
"""
Query the RAG pipeline — interactive REPL or single-shot mode.

Usage (interactive):
    python scripts/query.py --project my-project --dataset my_dataset

Usage (single query):
    python scripts/query.py --project my-project --dataset my_dataset \
        --query "What is the refund policy?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import VertexAIEmbedder
from src.vector_store import BigQueryVectorStore
from src.rag_chain import RAGChain


def print_result(result: dict, show_sources: bool = True) -> None:
    """Pretty-print a RAG result dict."""
    print(f"\nAnswer:\n{result['answer']}")
    if show_sources:
        print("\nSources retrieved:")
        for s in result["sources"]:
            print(f"  [{s['source']}]  cosine distance: {s['distance']:.6f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Vertex AI RAG pipeline")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--table", default="rag_documents", help="BigQuery table ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI region")
    parser.add_argument(
        "--credentials", default=None, help="Path to GCP service account JSON"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument("--query", help="Run a single query then exit")
    parser.add_argument(
        "--no-sources", action="store_true", help="Hide source citations"
    )
    args = parser.parse_args()

    print("Initialising RAG pipeline...")
    embedder = VertexAIEmbedder(project_id=args.project, location=args.location)
    vector_store = BigQueryVectorStore(
        project_id=args.project,
        dataset_id=args.dataset,
        table_id=args.table,
        credentials_path=args.credentials,
    )
    rag = RAGChain(
        embedder=embedder,
        vector_store=vector_store,
        project_id=args.project,
        location=args.location,
        top_k=args.top_k,
    )
    print("Ready.\n")

    # ── Single-shot mode ──────────────────────────────────────────────────────
    if args.query:
        result = rag.ask(args.query)
        print(f"Q: {args.query}")
        print_result(result, show_sources=not args.no_sources)
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("Ask questions about your documents. Press Ctrl+C or type 'quit' to exit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        try:
            result = rag.ask(question)
            print_result(result, show_sources=not args.no_sources)
        except Exception as exc:
            print(f"  Error: {exc}\n")


if __name__ == "__main__":
    main()
