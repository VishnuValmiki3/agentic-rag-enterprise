"""
Evaluation Metrics — measures retrieval quality and answer faithfulness.
This is the module that separates your project from every other RAG demo.
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    actual_answer: str
    source_pages: list[int]
    expected_pages: list[int]
    retrieval_hit: bool
    faithfulness_score: float
    citation_accuracy: float
    latency_ms: float
    retries: int


@dataclass
class EvalSummary:
    total_questions: int = 0
    retrieval_hit_rate: float = 0.0
    mean_faithfulness: float = 0.0
    mean_citation_accuracy: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_retries: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 3),
            "mean_faithfulness": round(self.mean_faithfulness, 3),
            "mean_citation_accuracy": round(self.mean_citation_accuracy, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "avg_retries": round(self.avg_retries, 2),
        }


def load_test_set(path: str = "src/evaluation/test_set.json") -> list[dict]:
    """Load the manually created test set."""
    with open(path) as f:
        return json.load(f)


def check_retrieval_hit(
    retrieved_pages: list[int],
    expected_pages: list[int],
) -> bool:
    """Did retrieval return at least one page from the expected set?"""
    return bool(set(retrieved_pages) & set(expected_pages))


def score_faithfulness(answer: str, source_texts: list[str], llm=None) -> float:
    """
    Score how faithful the answer is to the source documents.

    Simple heuristic version (no LLM needed):
    - Check what fraction of sentences in the answer can be traced to sources
    - Returns 0.0 to 1.0

    For production, use RAGAS or LLM-as-judge.
    """
    if not answer or not source_texts:
        return 0.0

    # Simple: check if answer sentences contain words from sources
    source_words = set()
    for text in source_texts:
        source_words.update(text.lower().split())

    answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not answer_sentences:
        return 0.0

    grounded_count = 0
    for sentence in answer_sentences:
        sentence_words = set(sentence.lower().split())
        # If >50% of sentence words appear in sources, consider it grounded
        if sentence_words and len(sentence_words & source_words) / len(sentence_words) > 0.5:
            grounded_count += 1

    return grounded_count / len(answer_sentences)


def score_citation_accuracy(
    answer: str,
    retrieved_pages: list[int],
) -> float:
    """Check if page citations in the answer match retrieved pages."""
    import re

    # Find all page citations in either format:
    #   [Page 5] or [Page 5, Section X]
    #   [Chunk N | Page 5 | Section: ...]
    cited_pages = []
    for match in re.finditer(r'Page\s+(\d+)', answer):
        cited_pages.append(int(match.group(1)))

    if not cited_pages:
        # No citations found — this is bad for enterprise use
        return 0.0

    # What fraction of cited pages were actually retrieved?
    correct = sum(1 for p in cited_pages if p in retrieved_pages)
    return correct / len(cited_pages)


def run_evaluation(agent, test_set: list[dict]) -> EvalSummary:
    """Run the full evaluation pipeline."""
    from ..agent.graph import query as run_query

    results = []
    latencies = []

    for i, test_case in enumerate(test_set):
        print(f"  Evaluating [{i+1}/{len(test_set)}]: {test_case['question'][:60]}...")

        start = time.time()
        result = run_query(agent, test_case["question"])
        elapsed_ms = (time.time() - start) * 1000

        retrieved_pages = [s["page"] for s in result.get("sources", []) if isinstance(s.get("page"), int)]
        expected_pages = test_case.get("expected_pages", [])
        source_texts = [s["text"] for s in result.get("sources", [])]

        eval_result = EvalResult(
            question=test_case["question"],
            expected_answer=test_case.get("expected_answer", ""),
            actual_answer=result["answer"],
            source_pages=retrieved_pages,
            expected_pages=expected_pages,
            retrieval_hit=check_retrieval_hit(retrieved_pages, expected_pages),
            faithfulness_score=score_faithfulness(result["answer"], source_texts),
            citation_accuracy=score_citation_accuracy(result["answer"], retrieved_pages),
            latency_ms=elapsed_ms,
            retries=result.get("retries", 0),
        )
        results.append(eval_result)
        latencies.append(elapsed_ms)

    # Compute summary
    latencies.sort()
    n = len(results)

    summary = EvalSummary(
        total_questions=n,
        retrieval_hit_rate=sum(r.retrieval_hit for r in results) / n if n else 0,
        mean_faithfulness=sum(r.faithfulness_score for r in results) / n if n else 0,
        mean_citation_accuracy=sum(r.citation_accuracy for r in results) / n if n else 0,
        latency_p50_ms=latencies[n // 2] if n else 0,
        latency_p95_ms=latencies[int(n * 0.95)] if n else 0,
        avg_retries=sum(r.retries for r in results) / n if n else 0,
        results=results,
    )

    return summary
