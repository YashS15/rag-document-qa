import json
from pathlib import Path

import faiss
import numpy as np

STORE_DIR = Path("vectorstore")
INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH = STORE_DIR / "metadata.json"
DIMENSION = 384  # all-MiniLM-L6-v2


class VectorStore:
    def __init__(self):
        STORE_DIR.mkdir(exist_ok=True)
        # chunks list mirrors FAISS index positions — never shrink it on delete
        self.chunks: list[dict] = []
        self.documents: dict[str, dict] = {}
        # IndexFlatIP with normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(DIMENSION)
        self._load()

    def _load(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH) as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.documents = data.get("documents", {})

    def _save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w") as f:
            json.dump({"chunks": self.chunks, "documents": self.documents}, f)

    def add_document(self, doc_meta: dict, chunks: list[dict], embeddings: np.ndarray):
        doc_id = doc_meta["doc_id"]
        self.documents[doc_id] = {
            "name": doc_meta["name"],
            "page_count": doc_meta["page_count"],
            "chunk_count": doc_meta["chunk_count"],
            "uploaded_at": doc_meta["uploaded_at"],
        }
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk["faiss_idx"] = start_idx + i
            self.chunks.append(chunk)
        self.index.add(embeddings)
        self._save()

    def search(self, query_embedding: np.ndarray, k: int = 5, doc_id: str | None = None) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_vec = query_embedding.reshape(1, -1)
        n = min(k * 4, self.index.ntotal)
        scores, indices = self.index.search(query_vec, n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if chunk["doc_id"] not in self.documents:
                continue
            if doc_id and chunk["doc_id"] != doc_id:
                continue  # filter to specific document
            results.append({**chunk, "score": float(score)})
            if len(results) >= k:
                break
        return results

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        # Chunks stay in self.chunks so FAISS indices remain valid;
        # search filters them out via the documents dict check above.
        self._save()
        return True

    def list_documents(self) -> list[dict]:
        return [{"doc_id": did, **meta} for did, meta in self.documents.items()]
