"""
Hybrid Retrieval — combines vector search + BM25 + cross-encoder re-ranking.
Uses Reciprocal Rank Fusion (RRF) to merge results from both sources.
"""
from .vector_store import VectorStore
from .bm25_search import BM25Search
from .reranker import Reranker


class HybridRetriever:
    """Two-stage hybrid retrieval: recall (vector+BM25) → precision (re-ranker)."""

    def __init__(
        self,
        vector_store: VectorStore,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        rerank_top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25 = BM25Search()
        self.reranker = Reranker()

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k

        # Build BM25 index from vector store contents
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the vector store."""
        all_docs = self.vector_store.get_all_documents()
        if all_docs:
            self.bm25.build_index(all_docs)

    def retrieve(self, query: str) -> list[dict]:
        """Full hybrid retrieval pipeline."""
        # Stage 1: Recall — get candidates from both sources
        vector_results = self.vector_store.search(query, top_k=self.vector_top_k)
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k)

        # Merge with Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Stage 2: Precision — re-rank top candidates
        reranked = self.reranker.rerank(query, fused, top_k=self.rerank_top_k)

        return reranked

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Merge results using Reciprocal Rank Fusion (RRF)."""
        # Track by text content (deduplicate)
        doc_scores = {}
        doc_map = {}

        # Score vector results
        for rank, doc in enumerate(vector_results):
            key = doc["text"][:200]  # use first 200 chars as key
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            key = doc["text"][:200]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        # Sort by fused score
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc["rrf_score"] = doc_scores[key]
            results.append(doc)

        return results
