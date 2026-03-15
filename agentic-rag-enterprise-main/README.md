# Agentic RAG — Document Q&A with Self-Correcting Retrieval

RAG system for querying PDF document collections. Returns cited answers with page numbers. When initial retrieval is poor, a LangGraph agent rewrites the query and retries before generating an answer or refusing.

Tested on 4 financial documents (RBI Annual Report 2023-24, Economic Survey chapters 1, 3, 14) — 447 pages, 1,872 chunks.

## Architecture

```
PDF Documents
    │
    ▼
┌─────────────────────────────┐
│   Document Ingestion        │  PyMuPDF — header/footer removal,
│   (parse, clean, metadata)  │  table extraction, page-level metadata
└─────────────┬───────────────┘
              │
              ▼
┌──────────────────┐  ┌──────────────────────┐
│ Section-Aware    │──▶│ Embedding + Indexing  │
│ Chunking         │  │ sentence-transformers │
│                  │  │ → FAISS (flat IP)     │
└──────────────────┘  └──────────┬────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────┐
              │   Hybrid Retrieval              │
              │   Vector + BM25 (RRF fusion)    │
              │   → Cross-Encoder re-ranking    │
              └───────────────┬─────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │   LangGraph Agent               │
              │                                 │
              │   retrieve → grade ──────────►  │
              │                 │   generate    │
              │                 │   rewrite+    │
              │                 └── retry (×2)  │
              │                     refuse      │
              └───────────────┬─────────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
          ┌──────────────┐    ┌──────────────────┐
          │ Cited Answer │    │ Eval Pipeline    │
          │ with page    │    │ hit rate,        │
          │ citations    │    │ faithfulness,    │
          └──────┬───────┘    │ citation acc.    │
                 │            └──────────────────┘
                 ▼
          ┌──────────────┐
          │ Streamlit UI │
          └──────────────┘
```

## Results

Evaluated on 15 Q&A pairs manually constructed from the document set.

| Metric | Value |
|--------|-------|
| Retrieval hit rate | 93.3% (14/15) |
| Citation accuracy | 100% |
| Faithfulness (word-overlap heuristic) | 0.63 / 1.00 |
| Latency p50 | ~15s |
| Latency p95 | ~44s (query rewrite triggered) |
| Avg query rewrites per question | 0.13 |

The one miss (gross capital formation) retrieved a related table from the adjacent page returning a different GCF figure. The answer was factually close but failed the page-match check.

**Chunking strategy comparison** — same 15 queries, top-5 retrieval, in-memory FAISS:

| Strategy | Hit Rate | Chunks |
|----------|----------|--------|
| Fixed-256 words | 73.3% | 994 |
| Fixed-512 words | 73.3% | 576 |
| Section-aware | 86.7% | 1,872 |

Fixed-size chunking produces equal results at both window sizes on this corpus, suggesting the bottleneck is semantic boundary alignment, not chunk length. Section-aware chunking gains 13.4 points by keeping paragraphs and tables intact. See `notebooks/chunking_analysis.ipynb` for the per-query breakdown.

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF parsing | PyMuPDF (`fitz`) |
| Chunking | Custom section-aware (regex header detection, table preservation) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — runs locally |
| Vector store | FAISS `IndexFlatIP`, persisted as binary + JSON sidecar |
| Keyword search | `rank_bm25` (BM25Okapi) |
| Result fusion | Reciprocal Rank Fusion |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent | LangGraph `StateGraph` |
| LLM | GPT-4o-mini via `langchain-openai` |
| UI | Streamlit |

## Quick Start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add OPENAI_API_KEY to .env

# Ingest PDFs
python -m src.ingestion.pipeline --pdf-dir data/sample_pdfs/

# Run the UI
streamlit run src/ui/app.py

# Run evaluation
python -m src.evaluation.run_eval
```

## Project Structure

```
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py      # PyMuPDF extraction, header/footer filtering
│   │   ├── chunker.py         # Section-aware chunker
│   │   └── pipeline.py        # parse → chunk → embed → store
│   ├── retrieval/
│   │   ├── vector_store.py    # FAISS index with sentence-transformers
│   │   ├── bm25_search.py     # BM25 keyword search
│   │   ├── reranker.py        # Cross-encoder re-ranking
│   │   └── hybrid.py          # RRF fusion + re-rank pipeline
│   ├── agent/
│   │   ├── graph.py           # LangGraph state machine
│   │   ├── nodes.py           # retrieve, grade, rewrite, generate, refuse
│   │   └── prompts.py         # Prompts for each node
│   ├── evaluation/
│   │   ├── test_set.json      # 15 Q&A pairs with expected pages
│   │   ├── metrics.py         # hit rate, faithfulness, citation accuracy
│   │   └── run_eval.py        # Evaluation runner, outputs JSON report
│   └── ui/
│       └── app.py             # Streamlit chat interface
├── notebooks/
│   ├── chunking_analysis.ipynb       # Chunking strategy comparison
│   └── run_chunking_analysis.py      # Standalone runner (no nbconvert needed)
├── configs/config.yaml
├── .env.example
└── requirements.txt
```

## Limitations & Future Work

- **Latency.** End-to-end is 10–15s per query at p50, and up to 44s when query rewriting triggers. The bottleneck is sequential LLM calls for per-chunk grading. Needs async grading calls and a query-level cache for repeated questions.

- **Faithfulness scoring is a heuristic.** The current metric counts word overlap between the answer and source chunks. It systematically underscores answers where the LLM paraphrases rather than quotes. Needs replacement with an LLM-as-judge or RAGAS faithfulness scorer.

- **Small test corpus.** Evaluated on 4 documents, 447 pages. Retrieval performance at 10k+ pages with heterogeneous document types (contracts, policies, transcripts) is untested. Chunk count and BM25 index rebuild cost will need attention at scale.

- **No authentication or multi-tenancy.** The vector store holds all documents in a single flat index. Running this for multiple users or document namespaces requires per-tenant index isolation, which FAISS flat indices don't support natively (would need Qdrant/Weaviate with metadata filtering).

- **Single-document retrieval only.** The agent has no mechanism to synthesize across documents or answer comparative questions ("how does the RBI report differ from the Economic Survey on inflation?"). Would need a multi-hop or decomposition strategy.

## License

MIT
