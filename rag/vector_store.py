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
        metadatas = [{"doc_id": c.doc_id, "doc_name": c.doc_name, "order": c.order, "language": c.language, "category": c.category, "start_char": c.start_char, "end_char": c.end_char} for c in chunks]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_neighbors(self, chunk: Chunk, neighbors_forward: int = 3) -> list[Chunk]:
        hits = self.search_by_metadata(
            where = {
                "$and": [
                    {"doc_id": chunk.doc_id},
                    {"order": {"$gte": chunk.order - 1}},
                    {"order": {"$lte": chunk.order + neighbors_forward}},
                ]
            }
        )

        chunks = []
        for h in hits:
            meta = h["metadata"]
            chunks.append(
                Chunk(
                    id=h["id"],
                    doc_id=meta["doc_id"],
                    doc_name=meta["doc_name"],
                    text=h["document"],
                    order=meta["order"],
                    language=meta.get("language"),
                    category=meta.get("category"),
                    start_char=meta.get("start_char"),
                    end_char=meta.get("end_char"),
                )
            )

        return sorted(chunks, key=lambda c: c.order)


    def query(self, query_embedding: list[float], n_results: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        filters = []

        lang = where.get("language")
        if lang:
            if lang == "mixed":
                # хотим: ru, en и mixed
                filters.append({"language": {"$in": ["ru", "en", "mixed"]}})
            else:
                # хотим: либо конкретный язык, либо mixed
                filters.append({
                    "$or": [
                        {"language": lang},
                        {"language": "mixed"},
                    ]
                })

        category = where.get("category")
        if category:
            filters.append({"category": category})

        if filters:
            if len(filters) == 1:
                kwargs["where"] = filters[0]
            else:
                kwargs["where"] = {"$and": filters}


        result = self._collection.query(
            **kwargs
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

    def search_by_metadata(
        self,
        where: dict[str, Any],
    ) -> list[dict[str, Any]]:

        result = self._collection.get(
            where=where,
        )

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        hits = []
        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                }
            )

        return hits


    def clear_index(self, condition: dict | None = None) -> None:
            if condition:
                self._collection.delete(where=condition)
                return

            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
