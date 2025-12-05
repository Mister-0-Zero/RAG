from __future__ import annotations

from typing import Any

from rag.config import RAGConfig
from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class DenseRetriever:
    def __init__(
        self,
        cfg: RAGConfig | None = None,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
    ) -> None:
        cfg = cfg or RAGConfig()
        embedder = embedder or EmbeddingModel(cfg=cfg)
        store = store or VectorStore(cfg=cfg)

        self._cfg = cfg
        self._embedder = embedder
        self._store = store

        self._chunks_by_id: dict[str, Chunk] = {}

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        if clear:
            self._store.clear()

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._store.index_chunks(chunks, embeddings)

        self._chunks_by_id = {c.id: c for c in chunks}

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_vec = self._embedder.embed_query(query)
        hits = self._store.query(q_vec, n_results=top_k)

        results: list[dict[str, Any]] = []

        for h in hits:
            chunk_id = h.get("id")
            distance = h.get("distance")

            chunk = self._chunks_by_id.get(chunk_id)

            results.append(
                {
                    "chunk": chunk,
                    "distance": distance,
                    "score": 1.0 - distance if distance is not None else None,
                }
            )

        return results
