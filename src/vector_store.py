"""BigQuery Vector Search — document storage, indexing, and retrieval."""

from __future__ import annotations

import json
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryVectorStore:
    """
    Stores document embeddings in BigQuery and retrieves them via
    the VECTOR_SEARCH function with cosine similarity and hybrid search.

    Table schema:
        id        STRING   — unique chunk ID
        content   STRING   — raw text of the chunk
        source    STRING   — originating file/document name
        metadata  STRING   — JSON blob for arbitrary extra fields
        embedding FLOAT64  REPEATED — dense embedding vector
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        credentials_path: str | None = None,
        location: str = "US",
    ):
        """
        Args:
            project_id: GCP project ID.
            dataset_id: BigQuery dataset ID.
            table_id: BigQuery table ID (will be created if absent).
            credentials_path: Path to service account JSON. Falls back to
                              GOOGLE_APPLICATION_CREDENTIALS env var.
            location: BigQuery dataset location (default US).
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        self.location = location

        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            self.client = bigquery.Client(project=project_id, credentials=credentials)
        else:
            self.client = bigquery.Client(project=project_id)

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_table(self) -> None:
        """Create the documents table if it doesn't already exist."""
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("metadata", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        table = bigquery.Table(self.full_table_id, schema=schema)
        self.client.create_table(table, exists_ok=True)
        print(f"  Table ready: {self.full_table_id}")

    def create_vector_index(self) -> None:
        """
        Create an IVF vector index on the embedding column for fast ANN search.

        Safe to call on an existing index (uses IF NOT EXISTS).
        Note: BigQuery requires ≥5,000 rows before the index is used.
        """
        index_name = f"{self.table_id}_embedding_idx"
        query = f"""
        CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
        ON `{self.full_table_id}`(embedding)
        OPTIONS (distance_type = 'COSINE', index_type = 'IVF')
        """
        self.client.query(query).result()
        print(f"  Vector index ready: {index_name}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_documents(self, documents: list[dict]) -> None:
        """
        Insert document chunks with their embeddings into BigQuery.

        Args:
            documents: List of dicts with keys:
                - content (str)
                - embedding (list[float])
                - source (str, optional)
                - metadata (dict, optional)
                - id (str, optional — auto-generated if absent)
        """
        rows = [
            {
                "id": doc.get("id", str(uuid.uuid4())),
                "content": doc["content"],
                "source": doc.get("source", ""),
                "metadata": json.dumps(doc.get("metadata", {})),
                "embedding": doc["embedding"],
            }
            for doc in documents
        ]
        errors = self.client.insert_rows_json(self.full_table_id, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k most semantically similar chunks via VECTOR_SEARCH.
        Uses safe BigQuery array parameters rather than raw string interpolation.

        Args:
            query_embedding: Dense embedding of the user query.
            top_k: Number of results to return.

        Returns:
            List of dicts: {id, content, source, metadata, distance}
            Ordered by ascending cosine distance (closer = more relevant).
        """
        query = f"""
        SELECT
            base.id,
            base.content,
            base.source,
            base.metadata,
            distance
        FROM
            VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                'embedding',
                (SELECT @query_embedding AS embedding),
                top_k          => @top_k,
                distance_type  => 'COSINE'
            )
        ORDER BY distance ASC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )
        results = self.client.query(query, job_config=job_config).result()
        return [
            {
                "id": row.id,
                "content": row.content,
                "source": row.source,
                "metadata": json.loads(row.metadata or "{}"),
                "distance": row.distance,
            }
            for row in results
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Dual-stage hybrid retrieval combining lexical keyword search and semantic
        vector search, unified via Reciprocal Rank Fusion (RRF, k=60).

        Args:
            query_text: The user's search query for lexical filtering.
            query_embedding: Dense embedding vector for semantic search.
            top_k: Final number of ranked results to return.
            rrf_k: Smoothing constant for reciprocal rank fusion (default 60).
        """
        # 1. Semantic vector candidates (fetch top_k * 2)
        semantic_results = self.similarity_search(query_embedding, top_k=top_k * 2)

        # 2. Lexical keyword candidates
        lexical_query = f"""
        SELECT
            id,
            content,
            source,
            metadata
        FROM
            `{self.full_table_id}`
        WHERE
            LOWER(content) LIKE CONCAT('%', LOWER(@term), '%')
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("term", "STRING", query_text),
                bigquery.ScalarQueryParameter("limit", "INT64", top_k * 2),
            ]
        )
        lexical_results = [
            {
                "id": row.id,
                "content": row.content,
                "source": row.source,
                "metadata": json.loads(row.metadata or "{}"),
                "distance": 0.5,
            }
            for row in self.client.query(lexical_query, job_config=job_config).result()
        ]

        # 3. Reciprocal Rank Fusion (RRF)
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, doc in enumerate(lexical_results, start=1):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Rank documents by descending RRF score
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        fused_results = []
        for doc_id in ranked_ids:
            item = dict(doc_map[doc_id])
            item["rrf_score"] = round(scores[doc_id], 6)
            fused_results.append(item)

        return fused_results

    def count_documents(self) -> int:
        """Return the number of chunks currently stored."""
        query = f"SELECT COUNT(*) AS n FROM `{self.full_table_id}`"
        result = self.client.query(query).result()
        return next(iter(result)).n
