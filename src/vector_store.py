"""BigQuery Vector Search — document storage, indexing, and retrieval."""

from __future__ import annotations

import json
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryVectorStore:
    """
    Stores document embeddings in BigQuery and retrieves them via
    the VECTOR_SEARCH function with cosine similarity.

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

        Args:
            query_embedding: Dense embedding of the user query.
            top_k: Number of results to return.

        Returns:
            List of dicts: {id, content, source, metadata, distance}
            Ordered by ascending cosine distance (closer = more relevant).
        """
        # BigQuery expects the query embedding as a subquery returning the same
        # column name as the indexed field.
        embedding_literal = json.dumps(query_embedding)
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
                (SELECT {embedding_literal} AS embedding),
                top_k          => {top_k},
                distance_type  => 'COSINE'
            )
        ORDER BY distance ASC
        """
        results = self.client.query(query).result()
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

    def count_documents(self) -> int:
        """Return the number of chunks currently stored."""
        query = f"SELECT COUNT(*) AS n FROM `{self.full_table_id}`"
        result = self.client.query(query).result()
        return next(iter(result)).n
