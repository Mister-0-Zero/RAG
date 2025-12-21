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

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        print("Построение индекса векторного хранилища...")

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

        hits = self._store.query(q_vec, n_results=top_k, where={
            "language": language,
            "category": category,
        })

        results: list[dict[str, Any]] = []

        for h in hits:
            chunk_id = h.get("id")
            distance = h.get("distance")

            meta = h.get("metadata", {})
            text = h.get("document")

            chunk = Chunk(
                id=chunk_id,
                doc_id=meta.get("doc_id"),
                doc_name=meta.get("doc_name", "unknown"),
                text=text,
                order=meta.get("order", 0),
                language=meta.get("language"),
                category=meta.get("category"),
                start_char=meta.get("start_char"),
                end_char=meta.get("end_char"),
            )

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
                    "score": 1.0 / (1.0 + distance) if distance is not None else None,
                    "metadata": h.get("metadata", {}),
                }
            )

        return results
