"""
Cross-Encoder Re-ranker — precision-optimized second stage.
Takes the top-20 candidates from hybrid search and picks the best 5.
"""
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class Reranker:
    """Cross-encoder re-ranking for precision retrieval."""

    def __init__(self, model_name: str = None):
        model_name = model_name or RERANKER_MODEL
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]
