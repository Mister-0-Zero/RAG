from __future__ import annotations

from typing import Any, List

from elasticsearch.helpers import bulk

from search.es_client import get_es
from rag.chunking import Chunk


class ElasticsearchLexicalRetriever:
    def __init__(self, index_name: str = "hd_chunks") -> None:
        self._es = get_es()
        self._index_name = index_name
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self._es.indices.exists(index=self._index_name):
            body = {
                "mappings": {
                    "properties": {
                        "id":        {"type": "keyword"},
                        "doc_id":    {"type": "keyword"},
                        "doc_name":  {"type": "keyword"},
                        "text":      {"type": "text"},
                        "order":     {"type": "integer"},
                        "language":  {"type": "keyword"},
                        "category":  {"type": "keyword"},
                        "start_char": {"type": "integer"},
                        "end_char":   {"type": "integer"},
                    }
                }
            }
            self._es.indices.create(index=self._index_name, body=body)

    def index_chunks(self, chunks: List[Chunk], clear: bool = True) -> None:
        if clear:
            self._es.indices.delete(index=self._index_name)
            self._ensure_index()

        actions = []
        for c in chunks:
            actions.append({
                "_index": self._index_name,
                "_id": c.id,
                "_source": {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "text": c.text,
                    "order": c.order,
                    "language": c.language or "mixed",
                    "category": c.category or "general",
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                },
            })

        bulk(self._es, actions)

    def clear_index(self) -> None:
        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)
        self._ensure_index()

    def search(self, query: str, top_k: int = 10, language: str | None = None, category: str | None = None) -> List[Any]:
        filters: list[dict[str, Any]] = []

        if language in ("ru", "en"):
            filters.append({
                "terms": {"language": [language, "mixed"]}
            })

        if category in ("gate", "channel", "center"):
            filters.append({"term": {"category": category}})


        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text"],
                        }
                    },
                    "filter": filters or [],
                }
            },
            "size": top_k,
        }

        resp = self._es.search(index=self._index_name, body=body)

        hits = resp["hits"]["hits"]

        results: list[dict[str, Any]] = []
        for h in hits:
            src = h["_source"]

            chunk = Chunk(
                id=src["id"],
                doc_id=src["doc_id"],
                doc_name=src["doc_name"],
                text=src["text"],
                order=src["order"],
                start_char=src.get("start_char", 0),
                end_char=src.get("end_char", 0),
                language=src.get("language"),
                category=src.get("category"),
            )

            results.append(
                {
                    "chunk": chunk,
                    "score": float(h["_score"]),
                }
            )

        return results
