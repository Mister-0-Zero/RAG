from __future__ import annotations

from typing import Any

import chromadb

from rag.config import RAGConfig
from rag.chunking import Chunk


class VectorStore:
    def __init__(
        self,
        cfg: RAGConfig | None = None,
        collection_name: str = "rag_chunks",
    ) -> None:
        if cfg is None:
            cfg = RAGConfig()

        self._cfg = cfg
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(cfg.chroma_db))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Количество чанков не равно количеству эмбеддингов!")

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [{"doc_id": c.doc_id, "order": c.order} for c in chunks]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )
        return hits

    def clear(self, condition: dict | None = None) -> None:
            if condition:
                self._collection.delete(where=condition)
                return
            
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
