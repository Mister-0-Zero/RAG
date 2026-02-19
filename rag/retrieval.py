"""
Provides a dense retriever for vector-based document search.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.config import RAGConfig
from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)


class DenseRetriever:
    """Manages the retrieval of documents based on dense vector similarity."""
    def __init__(
        self,
        cfg: RAGConfig | None = None,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """Initializes the DenseRetriever with a configuration, an embedding model, and a vector store."""
        cfg = cfg or RAGConfig()
        embedder = embedder or EmbeddingModel(cfg=cfg)
        store = store or VectorStore(cfg=cfg)

        self._cfg = cfg
        self._embedder = embedder
        self._store = store

        self._chunks_by_id: dict[str, Chunk] = {}
        self._chunks_by_doc: dict[str, list[Chunk]] = {}

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        """Builds the vector index from a list of chunks."""
        log.info(f"Building vector store index with {len(chunks)} chunks...", extra={'log_type': 'INFO'})

        if self._cfg.section_title_in_embeddings:
            texts = [
                f"{c.section_title}\n{c.text}" if c.section_title else c.text
                for c in chunks
            ]
        else:
            texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._store.index_chunks(chunks, embeddings)

        self._chunks_by_id = {c.id: c for c in chunks}

        self._chunks_by_doc = {}
        for c in chunks:
            self._chunks_by_doc.setdefault(c.doc_id, []).append(c)

        for doc_id, doc_chunks in self._chunks_by_doc.items():
            doc_chunks.sort(key=lambda x: x.order)
        
        log.info("Finished building vector store index.", extra={'log_type': 'INFO'})

    def retrieve(self, query: str, top_k: int = 5, language: str | None = None, category: str | None = None, neighbors: int = 0) -> list[dict[str, Any]]:
        """Retrieves relevant documents for a given query from the vector store."""
        q_vec = self._embedder.embed_query(query)
        
        where_filter = {"language": language, "category": category}
        log.info(f"Performing dense retrieval for query: '{query}' with filter: {where_filter}", extra={'log_type': 'INFO'})

        hits = self._store.query(q_vec, n_results=top_k, where=where_filter)
        log.info(f"Found {len(hits)} dense hits.", extra={'log_type': 'INFO'})

        results: list[dict[str, Any]] = []

        for h in hits:
            chunk_id = str(h.get("id") or "")
            distance = h.get("distance")

            meta = h.get("metadata", {})
            text = str(h.get("document") or "")

            chunk = Chunk(
                id=chunk_id,
                doc_id=meta.get("doc_id") or "",
                doc_name=meta.get("doc_name", "unknown"),
                text=text,
                order=meta.get("order", 0),
                section_title=meta.get("section_title"),
                language=meta.get("language"),
                category=meta.get("category"),
                start_char=meta.get("start_char"),
                end_char=meta.get("end_char"),
                allowed_roles=meta.get("allowed_roles"),
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
