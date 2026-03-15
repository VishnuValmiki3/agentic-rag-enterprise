"""Fact extraction helper — queries FAISS+BM25 to surface ground-truth content."""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search

store = VectorStore(persist_dir="./data/chroma_db")
bm25 = BM25Search()
bm25.build_index(store.get_all_documents())

queries = [
    ("GDP growth 2023-24", "gdp"),
    ("gross NPA ratio scheduled commercial banks", "npa"),
    ("CPI inflation headline 2023-24", "inflation"),
    ("repo rate monetary policy committee unchanged 6.50", "repo"),
    ("foreign exchange reserves end March 2024", "forex"),
    ("current account deficit 2023-24", "cad"),
    ("fiscal deficit GDP percentage 2023-24", "fiscal"),
    ("gross capital formation investment", "gcf"),
    ("bank credit growth 2023-24", "credit"),
    ("AI artificial intelligence policy India", "ai"),
    ("digital economy technology regulation", "digital"),
    ("agriculture sector growth rural", "agri"),
    ("India external debt short term", "extdebt"),
    ("remittances FPI FDI capital flows", "flows"),
    ("employment labour market jobs", "employment"),
]

for query, tag in queries:
    results = bm25.search(query, top_k=2)
    if not results:
        results = store.search(query, top_k=2)
    print(f"\n{'='*60}")
    print(f"[{tag.upper()}] {query}")
    for r in results[:2]:
        m = r["metadata"]
        print(f"  File: {m.get('filename','?')} | Page: {m.get('page_number','?')} | Section: {m.get('section','')[:50]}")
        print(f"  {r['text'][:400].strip()}")
