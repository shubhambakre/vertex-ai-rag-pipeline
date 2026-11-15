import json
import sys
from unittest.mock import MagicMock

# Mock google.cloud.bigquery before importing vector_store
mock_bigquery = MagicMock()
mock_google = MagicMock()
mock_google.cloud = MagicMock()
mock_google.cloud.bigquery = mock_bigquery
mock_oauth2 = MagicMock()

# Setup ArrayQueryParameter, ScalarQueryParameter, QueryJobConfig
class DummyArrayQueryParameter:
    def __init__(self, name, param_type, values):
        self.name = name
        self.param_type = param_type
        self.values = values

class DummyScalarQueryParameter:
    def __init__(self, name, param_type, value):
        self.name = name
        self.param_type = param_type
        self.value = value

class DummyQueryJobConfig:
    def __init__(self, query_parameters=None, **kwargs):
        self.query_parameters = query_parameters or []

mock_bigquery.ArrayQueryParameter = DummyArrayQueryParameter
mock_bigquery.ScalarQueryParameter = DummyScalarQueryParameter
mock_bigquery.QueryJobConfig = DummyQueryJobConfig

sys.modules["google"] = mock_google
sys.modules["google.cloud"] = mock_google.cloud
sys.modules["google.cloud.bigquery"] = mock_bigquery
sys.modules["google.oauth2"] = mock_oauth2
sys.modules["google.oauth2.service_account"] = MagicMock()

from src.vector_store import BigQueryVectorStore


class MockRow:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return self.__dict__.get(name)


def test_similarity_search_parameterization():
    mock_client = MagicMock()
    mock_query_job = MagicMock()
    mock_row = MockRow(
        id="doc-1",
        content="GCP BigQuery Vector Search",
        source="manual.pdf",
        metadata=json.dumps({"section": "overview"}),
        distance=0.15,
    )
    mock_query_job.result.return_value = [mock_row]
    mock_client.query.return_value = mock_query_job

    store = BigQueryVectorStore(
        project_id="test-proj",
        dataset_id="test_ds",
        table_id="test_table",
    )
    store.client = mock_client

    results = store.similarity_search([0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "doc-1"
    assert results[0]["distance"] == 0.15

    # Verify parameterized query was used
    call_args, call_kwargs = mock_client.query.call_args
    assert "@query_embedding" in call_args[0]
    job_config = call_kwargs["job_config"]
    assert len(job_config.query_parameters) == 2
    assert isinstance(job_config.query_parameters[0], DummyArrayQueryParameter)
    assert job_config.query_parameters[0].name == "query_embedding"


def test_hybrid_search_rrf():
    mock_client = MagicMock()

    # Semantic search results
    sem_row = MockRow(
        id="doc-sem",
        content="Semantic concept content",
        source="sem.pdf",
        metadata="{}",
        distance=0.1,
    )
    # Lexical search results
    lex_row = MockRow(
        id="doc-lex",
        content="Exact keyword match content",
        source="lex.pdf",
        metadata="{}",
    )

    mock_sem_job = MagicMock()
    mock_sem_job.result.return_value = [sem_row]

    mock_lex_job = MagicMock()
    mock_lex_job.result.return_value = [lex_row]

    mock_client.query.side_effect = [mock_sem_job, mock_lex_job]

    store = BigQueryVectorStore(
        project_id="test-proj",
        dataset_id="test_ds",
        table_id="test_table",
    )
    store.client = mock_client

    results = store.hybrid_search(
        query_text="exact keyword",
        query_embedding=[0.1, 0.2],
        top_k=2,
        rrf_k=60,
    )

    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "doc-sem" in ids
    assert "doc-lex" in ids
    for r in results:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0
