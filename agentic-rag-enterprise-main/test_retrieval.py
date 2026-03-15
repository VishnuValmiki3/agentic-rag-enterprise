"""
Quick smoke test: ingest PDFs and run sample queries to verify retrieval.
Run: python test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import run_ingestion
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search

PDF_DIR = "./data/sample_pdfs"
CHROMA_DIR = "./data/chroma_db"

SAMPLE_QUERIES = [
    "What is India's GDP growth rate?",
    "What are the key recommendations of the Economic Survey?",
    "RBI monetary policy and interest rates",
    "Inflation trends in India",
    "Foreign exchange reserves",
]


def run_ingestion_if_needed():
    store = VectorStore(persist_dir=CHROMA_DIR)
    if store.count > 0:
        print(f"Vector store already has {store.count} chunks — skipping ingestion.\n")
        return store
    print("No existing index found — running ingestion...\n")
    run_ingestion(PDF_DIR, CHROMA_DIR)
    return VectorStore(persist_dir=CHROMA_DIR)


def test_vector_search(store: VectorStore):
    print("=" * 60)
    print("VECTOR SEARCH TEST")
    print("=" * 60)
    for query in SAMPLE_QUERIES:
        results = store.search(query, top_k=3)
        print(f"\nQuery: {query}")
        if results:
            top = results[0]
            meta = top["metadata"]
            print(f"  Score: {top['score']:.4f} | File: {meta.get('filename', '?')} | Page: {meta.get('page_number', '?')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Text: {top['text'][:200].strip()}...")
        else:
            print("  No results found.")


def test_bm25_search(store: VectorStore):
    print("\n" + "=" * 60)
    print("BM25 KEYWORD SEARCH TEST")
    print("=" * 60)
    bm25 = BM25Search()
    all_docs = store.get_all_documents()
    bm25.build_index(all_docs)

    keyword_queries = ["GDP growth", "repo rate", "inflation", "current account deficit", "fiscal deficit"]
    for query in keyword_queries:
        results = bm25.search(query, top_k=3)
        print(f"\nQuery: {query}")
        if results:
            top = results[0]
            meta = top["metadata"]
            print(f"  BM25 Score: {top['score']:.4f} | File: {meta.get('filename', '?')} | Page: {meta.get('page_number', '?')}")
            print(f"  Text: {top['text'][:200].strip()}...")
        else:
            print("  No results found.")


if __name__ == "__main__":
    store = run_ingestion_if_needed()
    print(f"\nTotal chunks indexed: {store.count}\n")
    test_vector_search(store)
    test_bm25_search(store)
    print("\n" + "=" * 60)
    print("DONE — retrieval pipeline is working!")
    print("=" * 60)
