# vertex-ai-rag-pipeline

A Retrieval-Augmented Generation (RAG) pipeline on Google Cloud Platform. Ingests documents into **BigQuery Vector Search**, embeds queries with **Vertex AI text-embedding-004**, retrieves semantically relevant chunks, and generates grounded answers with **Gemini 1.5 Pro**.

No external vector database required — BigQuery handles both structured data and vector search in the same platform.

---

## Architecture

```
  Ingest                              Query
  ──────                              ─────

  Documents                           User Question
      │                                    │
      ▼                                    ▼
  ┌───────────────────┐            ┌───────────────────┐
  │  Text Chunker     │            │  VertexAIEmbedder │
  │  (word-level,     │            │  text-embedding-  │
  │   sliding window) │            │  004              │
  └────────┬──────────┘            │  RETRIEVAL_QUERY  │
           │                       └────────┬──────────┘
           ▼                                │
  ┌───────────────────┐                     ▼
  │  VertexAIEmbedder │            ┌───────────────────┐
  │  text-embedding-  │            │  BigQueryVector   │
  │  004              │            │  Store            │
  │  RETRIEVAL_DOC    │            │  VECTOR_SEARCH    │
  └────────┬──────────┘            │  (cosine, top-k)  │
           │                       └────────┬──────────┘
           ▼                                │
  ┌───────────────────┐                     ▼
  │  BigQueryVector   │            ┌───────────────────┐
  │  Store            │            │  RAGChain         │
  │  insert_rows_json │            │  Gemini 1.5 Pro   │
  │  + IVF index      │            │  grounded answer  │
  └───────────────────┘            └───────────────────┘
```

**Ingest pipeline** (`scripts/ingest.py`):
1. Split documents into overlapping word-level chunks (default 400 words, 50-word overlap)
2. Embed each chunk via Vertex AI `text-embedding-004` with `RETRIEVAL_DOCUMENT` task type
3. Store chunk text + embedding in BigQuery; build an IVF vector index for ANN retrieval

**Query pipeline** (`scripts/query.py` → `src/rag_chain.py`):
1. Embed the user question with `RETRIEVAL_QUERY` task type (asymmetric from document embeddings)
2. Run `VECTOR_SEARCH` in BigQuery — cosine similarity, returns top-k chunks
3. Assemble a grounded prompt with retrieved context and call Gemini 1.5 Pro
4. Return the answer + source citations + cosine distances for transparency

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | Vertex AI `text-embedding-004` |
| Vector store | Google BigQuery Vector Search (IVF index, cosine similarity) |
| Generation | Vertex AI Gemini 1.5 Pro |
| Auth | GCP Service Account JSON |
| Language | Python 3.11+ |

---

## Prerequisites

- Python 3.11+
- Google Cloud project with **Vertex AI API** and **BigQuery API** enabled
- Service account with roles: `BigQuery Data Editor`, `BigQuery Job User`, `Vertex AI User`
- BigQuery dataset created in your project

---

## Setup

```bash
git clone https://github.com/shubhambakre/vertex-ai-rag-pipeline.git
cd vertex-ai-rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Configure credentials:

```bash
cp .env.example .env
# Edit .env with your GCP_PROJECT_ID, BQ_DATASET_ID, GOOGLE_APPLICATION_CREDENTIALS
```

---

## Ingest Documents

```bash
python scripts/ingest.py \
    --project  your-gcp-project \
    --dataset  your_dataset \
    --files    docs/manual.txt docs/faq.txt docs/policy.txt
```

Options: `--chunk-size 400`, `--overlap 50`, `--table rag_documents`, `--credentials /path/to/sa.json`

The script creates the BigQuery table and IVF vector index on first run, then chunks, embeds, and stores the documents. Output:

```
Initialising embedder...
Initialising BigQuery vector store...
  Table ready: your-project.your_dataset.rag_documents
  Vector index ready: rag_documents_embedding_idx

Ingesting: docs/manual.txt
  manual.txt: 42 chunks
    Embedded batch 1/9
    ...
  ✅  42 chunks stored from 'manual.txt'

🎉  Done. 42 new chunks ingested. 42 total in store.
```

---

## Query

**Interactive REPL:**
```bash
python scripts/query.py --project your-gcp-project --dataset your_dataset
```
```
Initialising RAG pipeline...
Ready.

Ask questions about your documents. Press Ctrl+C or type 'quit' to exit.

Q: What is the return policy for electronics?

Answer:
According to the policy document, electronics can be returned within 30 days of
purchase with original packaging. Items must be in original condition. Opened
software and digital downloads are non-refundable.

Sources retrieved:
  [policy.txt]  cosine distance: 0.082341
  [policy.txt]  cosine distance: 0.094812
  [faq.txt]     cosine distance: 0.113205
```

**Single query:**
```bash
python scripts/query.py \
    --project your-gcp-project \
    --dataset your_dataset \
    --query "What is the return policy for electronics?"
```

---

## Project Structure

```
vertex-ai-rag-pipeline/
├── config.py               # Config dataclass loaded from .env
├── requirements.txt
├── .env.example            # Environment variable template
├── src/
│   ├── embedder.py         # VertexAIEmbedder — text-embedding-004, batched
│   ├── vector_store.py     # BigQueryVectorStore — table, IVF index, VECTOR_SEARCH
│   └── rag_chain.py        # RAGChain — retrieve → generate with Gemini 1.5 Pro
└── scripts/
    ├── ingest.py           # Chunk, embed, and store documents
    └── query.py            # Interactive REPL + single-shot query mode
```

---

## Design Notes

**Why BigQuery for vectors?** In production GCP environments, keeping vectors in BigQuery eliminates a separate vector database, simplifies IAM, and lets you join semantic search results with structured data (e.g. filter by date, category, or user permissions) in a single SQL query.

**Asymmetric embedding** (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`): Vertex AI's `text-embedding-004` is trained with separate task types for documents and queries. Using the correct task type per call materially improves retrieval precision.

**IVF index**: BigQuery's vector index is approximate (ANN) and activates above 5,000 rows. Below that threshold, `VECTOR_SEARCH` falls back to exact search automatically — no code change needed.

---

*Stack: Python · Vertex AI · Gemini 1.5 Pro · Google BigQuery · GCP*
