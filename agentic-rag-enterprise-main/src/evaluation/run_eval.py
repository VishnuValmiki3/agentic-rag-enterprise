"""
Evaluation runner — runs all test cases and prints a report.
Run: python -m src.evaluation.run_eval
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.agent.graph import build_agent
from src.evaluation.metrics import load_test_set, run_evaluation


def main():
    print("=" * 60)
    print("AGENTIC RAG EVALUATION PIPELINE")
    print("=" * 60)

    print("\nLoading system...")
    store = VectorStore(persist_dir="./data/chroma_db")
    print(f"  {store.count} chunks indexed")
    retriever = HybridRetriever(vector_store=store)
    agent = build_agent(retriever)

    print("\nLoading test set...")
    test_set = load_test_set("src/evaluation/test_set.json")
    print(f"  {len(test_set)} test cases")

    print("\nRunning evaluation...")
    summary = run_evaluation(agent, test_set)

    # Print per-question results
    print("\n" + "=" * 60)
    print("PER-QUESTION RESULTS")
    print("=" * 60)
    for r in summary.results:
        hit = "HIT " if r.retrieval_hit else "MISS"
        print(f"\n[{hit}] {r.question[:70]}")
        print(f"  Retrieved pages: {r.source_pages} | Expected: {r.expected_pages}")
        print(f"  Faithfulness: {r.faithfulness_score:.2f} | Citation accuracy: {r.citation_accuracy:.2f} | Latency: {r.latency_ms:.0f}ms | Retries: {r.retries}")
        print(f"  Answer: {r.actual_answer[:200].strip()}...")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = summary.to_dict()
    print(f"  Total questions    : {s['total_questions']}")
    print(f"  Retrieval hit rate : {s['retrieval_hit_rate']:.1%}")
    print(f"  Mean faithfulness  : {s['mean_faithfulness']:.2f} / 1.00")
    print(f"  Citation accuracy  : {s['mean_citation_accuracy']:.2f} / 1.00")
    print(f"  Latency p50        : {s['latency_p50_ms']:.0f}ms")
    print(f"  Latency p95        : {s['latency_p95_ms']:.0f}ms")
    print(f"  Avg retries        : {s['avg_retries']:.2f}")

    # Save results
    out_path = Path("data/eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": s,
            "results": [
                {
                    "question": r.question,
                    "retrieval_hit": r.retrieval_hit,
                    "faithfulness_score": round(r.faithfulness_score, 3),
                    "citation_accuracy": round(r.citation_accuracy, 3),
                    "latency_ms": round(r.latency_ms),
                    "retries": r.retries,
                    "source_pages": r.source_pages,
                    "expected_pages": r.expected_pages,
                    "actual_answer": r.actual_answer,
                }
                for r in summary.results
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
