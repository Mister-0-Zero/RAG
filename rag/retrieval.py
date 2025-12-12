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
        self._chunks_by_doc: dict[str, list[Chunk]] = {}

    def get_neighbors(self, chunk: Chunk, neighbors: int = 0) -> list[Chunk]:
        if neighbors <= 0:
            return [chunk]
        doc_chunks = self._chunks_by_doc.get(chunk.doc_id, [])
        if not doc_chunks:
            return [chunk]
        idx = chunk.order
        start = idx - 1 if idx - 1 >= 0 else 0
        end = min(len(doc_chunks), idx + neighbors + 1)
        return doc_chunks[start:end]

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        print("Построение индекса векторного хранилища...")
        if clear:
            self._store.clear()

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._store.index_chunks(chunks, embeddings)

        self._chunks_by_id = {c.id: c for c in chunks}

        self._chunks_by_doc = {}
        for c in chunks:
            self._chunks_by_doc.setdefault(c.doc_id, []).append(c)

        for doc_id, doc_chunks in self._chunks_by_doc.items():
            doc_chunks.sort(key=lambda x: x.order)


    def retrieve(self, query: str, top_k: int = 5, language: str | None = None, category: str | None = None, neighbors: int = 0) -> list[dict[str, Any]]:
        q_vec = self._embedder.embed_query(query)

        where: dict[str, Any] = {}
        if language:
            where["language"] = language
        if category:
            where["category"] = category

        if not where:
            where = None

        hits = self._store.query(q_vec, n_results=top_k, where=where)

        results: list[dict[str, Any]] = []

        for h in hits:
            chunk_id = h.get("id")
            distance = h.get("distance")

            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue

            idx = chunk.order
            if neighbors > 0:
                start = max(0, idx - 1)
                end = idx + neighbors + 1
                context_chunks = self._chunks_by_doc.get(chunk.doc_id, [])[start:end]
            else:
                context_chunks = [chunk]


            results.append(
                {
                    "chunk": context_chunks,
                    "main_chunk": chunk,
                    "distance": distance,
                    "score": 1.0 - distance if distance is not None else None,
                    "metadata": h.get("metadata", {}),
                }
            )

        return results
