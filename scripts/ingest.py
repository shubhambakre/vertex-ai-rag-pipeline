#!/usr/bin/env python3
"""
Ingest text documents into the BigQuery vector store.

Usage:
    python scripts/ingest.py --project my-project --dataset my_dataset \
        --files docs/manual.txt docs/faq.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import VertexAIEmbedder
from src.vector_store import BigQueryVectorStore


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text: Raw document text.
        chunk_size: Target chunk size in words.
        overlap: Number of words shared between adjacent chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ── Per-file ingestion ────────────────────────────────────────────────────────

def ingest_file(
    filepath: Path,
    embedder: VertexAIEmbedder,
    vector_store: BigQueryVectorStore,
    chunk_size: int,
    overlap: int,
) -> int:
    """Chunk, embed, and store a single file. Returns the number of chunks ingested."""
    text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"  {filepath.name}: {len(chunks)} chunks")

    documents: list[dict] = []
    total_batches = (len(chunks) + embedder._BATCH_SIZE - 1) // embedder._BATCH_SIZE

    for batch_idx in range(0, len(chunks), embedder._BATCH_SIZE):
        batch_texts = chunks[batch_idx : batch_idx + embedder._BATCH_SIZE]
        embeddings = embedder.embed_documents(batch_texts)

        for j, (chunk_text_str, embedding) in enumerate(zip(batch_texts, embeddings)):
            chunk_idx = batch_idx + j
            doc_id = hashlib.sha256(
                f"{filepath.name}:{chunk_idx}:{chunk_text_str[:64]}".encode()
            ).hexdigest()[:16]
            documents.append(
                {
                    "id": doc_id,
                    "content": chunk_text_str,
                    "source": filepath.name,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "file_path": str(filepath),
                    },
                }
            )

        batch_num = batch_idx // embedder._BATCH_SIZE + 1
        print(f"    Embedded batch {batch_num}/{total_batches}")

    vector_store.upsert_documents(documents)
    print(f"  ✅  {len(documents)} chunks stored from '{filepath.name}'")
    return len(documents)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into BigQuery Vector Store"
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--table", default="rag_documents", help="BigQuery table ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI region")
    parser.add_argument(
        "--bq-location", default="US", help="BigQuery dataset location"
    )
    parser.add_argument(
        "--credentials", default=None, help="Path to GCP service account JSON"
    )
    parser.add_argument(
        "--files", nargs="+", required=True, help="Text files to ingest"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=400, help="Chunk size in words"
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap in words between chunks"
    )
    args = parser.parse_args()

    print("Initialising embedder...")
    embedder = VertexAIEmbedder(project_id=args.project, location=args.location)

    print("Initialising BigQuery vector store...")
    vector_store = BigQueryVectorStore(
        project_id=args.project,
        dataset_id=args.dataset,
        table_id=args.table,
        credentials_path=args.credentials,
        location=args.bq_location,
    )
    vector_store.create_table()
    vector_store.create_vector_index()

    total_chunks = 0
    for filepath_str in args.files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"  ⚠️  File not found, skipping: {filepath}")
            continue
        print(f"\nIngesting: {filepath}")
        total_chunks += ingest_file(
            filepath, embedder, vector_store,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    doc_count = vector_store.count_documents()
    print(f"\n🎉  Done. {total_chunks} new chunks ingested. {doc_count} total in store.")


if __name__ == "__main__":
    main()
