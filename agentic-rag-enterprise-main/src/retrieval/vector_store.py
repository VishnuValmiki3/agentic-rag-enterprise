"""
Vector Store — FAISS-backed with sentence-transformers embeddings.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class VectorStore:
    """FAISS-backed vector store with local embeddings."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "enterprise_docs",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._documents: list[dict] = []
        self._index = None
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    def _index_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.collection_name}.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.collection_name}.json")

    def _load(self):
        if os.path.exists(self._index_path()) and os.path.exists(self._meta_path()):
            self._index = faiss.read_index(self._index_path())
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                self._documents = json.load(f)

    def _save(self):
        if self._index is not None:
            faiss.write_index(self._index, self._index_path())
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False)

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        batch_size: int = 64,
    ):
        """Add documents to the vector store in batches."""
        existing_ids = {doc["id"] for doc in self._documents}

        new_texts, new_metas, new_ids = [], [], []
        for text, meta, doc_id in zip(texts, metadatas, ids):
            if doc_id not in existing_ids:
                new_texts.append(text)
                new_metas.append(meta)
                new_ids.append(doc_id)
                existing_ids.add(doc_id)

        if not new_texts:
            return

        batches = []
        for i in range(0, len(new_texts), batch_size):
            emb = self.model.encode(new_texts[i : i + batch_size], show_progress_bar=False).astype(np.float32)
            faiss.normalize_L2(emb)
            batches.append(emb)

        emb_array = np.vstack(batches)

        if self._index is None:
            self._index = faiss.IndexFlatIP(emb_array.shape[1])
        self._index.add(emb_array)

        for text, meta, doc_id in zip(new_texts, new_metas, new_ids):
            self._documents.append({"text": text, "metadata": meta, "id": doc_id})

        self._save()

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search for similar documents."""
        if self._index is None or not self._documents:
            return []

        query_emb = self.model.encode([query], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(query_emb)

        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_emb, k)

        output = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            output.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score),  # cosine similarity (normalized dot product)
                "source": "vector",
            })
        return output

    def get_all_documents(self) -> list[dict]:
        """Get all documents (for BM25 index building)."""
        return [
            {"text": doc["text"], "metadata": doc["metadata"], "id": doc["id"]}
            for doc in self._documents
        ]

    @property
    def count(self) -> int:
        return len(self._documents)