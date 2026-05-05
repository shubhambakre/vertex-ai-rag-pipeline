"""Load configuration from environment variables or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    project_id: str
    dataset_id: str
    table_id: str
    location: str
    bq_location: str
    credentials_path: str | None
    top_k: int

    @classmethod
    def from_env(cls) -> "Config":
        required = ["GCP_PROJECT_ID", "BQ_DATASET_ID"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your values."
            )
        return cls(
            project_id=os.environ["GCP_PROJECT_ID"],
            dataset_id=os.environ["BQ_DATASET_ID"],
            table_id=os.getenv("BQ_TABLE_ID", "rag_documents"),
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            bq_location=os.getenv("BQ_LOCATION", "US"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
        )
