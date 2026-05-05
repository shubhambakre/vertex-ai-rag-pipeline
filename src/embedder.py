"""Document and query embedding using Vertex AI text-embedding-004."""

from __future__ import annotations

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class VertexAIEmbedder:
    """Wraps Vertex AI text-embedding-004 for document and query embedding."""

    # Maximum texts per API call (Vertex AI limit)
    _BATCH_SIZE = 5

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "text-embedding-004",
    ):
        """
        Args:
            project_id: GCP project ID.
            location: Vertex AI region (default us-central1).
            model_name: Embedding model to use.
        """
        vertexai.init(project=project_id, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts for storage/indexing.

        Automatically batches requests to respect Vertex AI rate limits.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
            embeddings = self.model.get_embeddings(inputs)
            all_embeddings.extend(e.values for e in embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Uses RETRIEVAL_QUERY task type — distinct from document embeddings,
        which optimises dot-product similarity between asymmetric query/doc pairs.

        Args:
            query: The user's natural language question.

        Returns:
            Embedding vector as a list of floats.
        """
        inputs = [TextEmbeddingInput(query, "RETRIEVAL_QUERY")]
        embeddings = self.model.get_embeddings(inputs)
        return embeddings[0].values
