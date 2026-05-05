"""RAG chain: retrieve relevant chunks, generate a grounded answer with Gemini."""

from __future__ import annotations

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from .embedder import VertexAIEmbedder
from .vector_store import BigQueryVectorStore


_SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions \
strictly based on the provided context documents.

Rules:
- Answer only from the context. Do not use prior knowledge.
- If the context doesn't contain enough information, say so explicitly.
- Cite the source document(s) when relevant.
- Be concise but complete."""

_USER_TEMPLATE = """\
Context documents:
{context}

---
Question: {question}

Answer:"""


class RAGChain:
    """
    End-to-end RAG pipeline:
      1. Embed the user query (Vertex AI text-embedding-004)
      2. Retrieve top-k semantically similar chunks (BigQuery VECTOR_SEARCH)
      3. Generate a grounded answer (Gemini 1.5 Pro)
    """

    def __init__(
        self,
        embedder: VertexAIEmbedder,
        vector_store: BigQueryVectorStore,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro",
        top_k: int = 5,
        temperature: float = 0.0,
    ):
        """
        Args:
            embedder: Initialised VertexAIEmbedder instance.
            vector_store: Initialised BigQueryVectorStore instance.
            project_id: GCP project ID (for Vertex AI generation).
            location: Vertex AI region.
            model_name: Gemini model to use for answer generation.
            top_k: Number of chunks to retrieve per query.
            temperature: Generation temperature (0.0 = deterministic).
        """
        vertexai.init(project=project_id, location=location)
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.model = GenerativeModel(
            model_name,
            system_instruction=_SYSTEM_PROMPT,
        )
        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )

    # ── Core steps ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query and fetch the top-k closest document chunks."""
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_embedding, top_k=self.top_k)

    def generate(self, question: str, context_docs: list[dict]) -> str:
        """Build a grounded prompt from retrieved docs and call Gemini."""
        context_blocks = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("source", "unknown")
            context_blocks.append(
                f"[{i}] Source: {source}\n{doc['content']}"
            )
        context = "\n\n".join(context_blocks)

        prompt = _USER_TEMPLATE.format(context=context, question=question)
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config,
        )
        return response.text

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a single question.

        Args:
            question: Natural language question from the user.

        Returns:
            Dict with keys:
                question    — the original question
                answer      — Gemini's grounded answer
                sources     — list of {id, source, distance} for each retrieved chunk
                context_docs — full retrieved chunk objects (for inspection/debugging)
        """
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "id": d["id"],
                    "source": d["source"],
                    "distance": round(d["distance"], 6),
                }
                for d in docs
            ],
            "context_docs": docs,
        }
