"""
Provides the main pipeline function for building the hybrid retriever.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rag.config import RAGConfig
from rag.ingest import ingest_all
from rag.chunking import chunk_documents
from rag.hybrid import HybridRetriever
from rag.retrieval import DenseRetriever
from rag.lexical_es import ElasticsearchLexicalRetriever
from rag.vector_store import VectorStore

log = logging.getLogger(__name__)

def build_hybrid_retriever(
    cfg: RAGConfig | None = None,
    chunk_size: int = 800,
    overlap: int = 200,
    reindex: bool = False,
) -> HybridRetriever:
    """
    Constructs and returns a `HybridRetriever`, optionally re-indexing the data.
    """
    cfg = cfg or RAGConfig()

    dense = DenseRetriever(cfg=cfg)
    lexical = ElasticsearchLexicalRetriever(index_name="hd_chunks", cfg=cfg)

    if reindex:
        log.info("Reindex enabled: rebuilding indexes...", extra={'log_type': 'INFO'})

        documents = ingest_all(cfg=cfg)
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
            cfg=cfg,
        )

        dense._store.clear_index()
        lexical.clear_index()

        hybrid = HybridRetriever(dense=dense, lexical=lexical, alpha=0.6)
        hybrid.build_index(chunks)

        log.info("Indexing finished.", extra={'log_type': 'INFO'})
        return hybrid

    log.info("Using existing indexes.", extra={'log_type': 'INFO'})
    return HybridRetriever(dense=dense, lexical=lexical, alpha=0.6)

