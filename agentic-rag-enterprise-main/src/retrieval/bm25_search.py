"""
BM25 Keyword Search — catches exact matches that embeddings miss.
Critical for policy numbers, names, dates, and specific terms.
"""
from rank_bm25 import BM25Okapi
import re


class BM25Search:
    """BM25 keyword search over document chunks."""

    def __init__(self):
        self.index = None
        self.documents = []

    def build_index(self, documents: list[dict]):
        """Build BM25 index from documents."""
        self.documents = documents
        tokenized = [self._tokenize(doc["text"]) for doc in documents]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search using BM25."""
        if self.index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[idx]),
                    "source": "bm25",
                })
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
