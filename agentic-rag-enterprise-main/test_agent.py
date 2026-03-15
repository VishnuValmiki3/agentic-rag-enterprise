"""
CLI smoke test for the full agentic RAG pipeline.
Run: python test_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.agent.graph import build_agent, query

CHROMA_DIR = "./data/chroma_db"

TEST_QUESTIONS = [
    "What was India's GDP growth in 2023-24?",
    "What is the repo rate set by the RBI?",
]


def main():
    print("Loading vector store...")
    vector_store = VectorStore(persist_dir=CHROMA_DIR)
    print(f"  {vector_store.count} chunks loaded")

    print("Building hybrid retriever (downloads cross-encoder on first run)...")
    retriever = HybridRetriever(vector_store=vector_store)

    print("Building LangGraph agent...")
    agent = build_agent(retriever)

    print("\n" + "=" * 60)
    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = query(agent, question)
        print(f"A: {result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for src in result["sources"]:
            print(f"  - Page {src['page']} | {src['section'][:60]} | score={src['score']:.3f}")
        latency = result["latency"]
        print(f"\nLatency: retrieval={latency.get('retrieval', 0)*1000:.0f}ms | "
              f"grading={latency.get('grading', 0)*1000:.0f}ms | "
              f"generation={latency.get('generation', 0)*1000:.0f}ms | "
              f"total={latency.get('total', 0)*1000:.0f}ms")
        if result.get("retries", 0):
            print(f"Query rewritten {result['retries']} time(s)")
        print("=" * 60)


if __name__ == "__main__":
    main()
