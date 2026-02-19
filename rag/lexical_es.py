"""
Provides an Elasticsearch-based lexical retriever.
"""
from __future__ import annotations

import logging
from typing import Any, List

from elasticsearch.helpers import bulk

from search.es_client import get_es
from rag.chunking import Chunk
from rag.config import RAGConfig

log = logging.getLogger(__name__)


class ElasticsearchLexicalRetriever:
    """Manages indexing and searching text chunks in Elasticsearch."""
    def __init__(self, index_name: str = "hd_chunks", cfg: RAGConfig | None = None) -> None:
        """Initializes the Elasticsearch client and ensures the index exists."""
        self._cfg = cfg or RAGConfig()
        self._es = get_es()
        self._index_name = index_name
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Creates the Elasticsearch index with the correct mapping if it doesn't exist."""
        if not self._es.indices.exists(index=self._index_name):
            log.info(f"Creating Elasticsearch index: {self._index_name}", extra={'log_type': 'INFO'})
            body = {
                "mappings": {
                    "properties": {
                        "id":        {"type": "keyword"},
                        "doc_id":    {"type": "keyword"},
                        "doc_name":  {"type": "keyword"},
                        "text":      {"type": "text"},
                        "section_title": {"type": "text"},
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
        """Indexes a list of chunks into Elasticsearch."""
        if clear:
            self.clear_index()

        log.info(f"Indexing {len(chunks)} chunks into Elasticsearch...", extra={'log_type': 'INFO'})
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
                    "section_title": c.section_title,
                    "order": c.order,
                    "language": c.language or "mixed",
                    "category": c.category or "general",
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                },
            })

        bulk(self._es, actions)
        log.info("Finished indexing.", extra={'log_type': 'INFO'})

    def clear_index(self) -> None:
        """Deletes and recreates the Elasticsearch index."""
        log.info(f"Clearing Elasticsearch index: {self._index_name}", extra={'log_type': 'INFO'})
        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)
        self._ensure_index()

    def search(self, query: str, top_k: int = 10, language: str | None = None, category: str | None = None) -> List[Any]:
        """Performs a lexical search against the Elasticsearch index."""
        filters: list[dict[str, Any]] = []

        if language in ("ru", "en"):
            filters.append({
                "terms": {"language": [language, "mixed"]}
            })

        if category in ("gate", "channel", "center"):
            filters.append({"term": {"category": category}})
        
        log.info(f"Performing lexical search for query: '{query}' with filters: {filters}", extra={'log_type': 'INFO'})

        fields = ["text"]
        if self._cfg.section_title_boost and self._cfg.section_title_boost > 0:
            fields.append(f"section_title^{self._cfg.section_title_boost}")

        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": fields,
                        }
                    },
                    "filter": filters or [],
                }
            },
            "size": top_k,
        }

        resp = self._es.search(index=self._index_name, body=body)

        hits = resp["hits"]["hits"]
        log.info(f"Found {len(hits)} lexical hits.", extra={'log_type': 'INFO'})

        results: list[dict[str, Any]] = []
        for h in hits:
            src = h["_source"]

            chunk = Chunk(
                id=src["id"],
                doc_id=src["doc_id"],
                doc_name=src["doc_name"],
                text=src["text"],
                order=src["order"],
                section_title=src.get("section_title"),
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
