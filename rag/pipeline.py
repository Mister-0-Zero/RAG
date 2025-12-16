from __future__ import annotations

from pathlib import Path

from rag.config import RAGConfig
from rag.ingest import ingest_all
from rag.chunking import chunk_documents
from rag.hybrid import HybridRetriever
from rag.retrieval import DenseRetriever
from rag.lexical_es import ElasticsearchLexicalRetriever
from rag.vector_store import VectorStore

def build_hybrid_retriever(
    cfg: RAGConfig | None = None,
    chunk_size: int = 800,
    overlap: int = 200,
    reindex: bool = False,
) -> HybridRetriever:

    cfg = cfg or RAGConfig()

    dense = DenseRetriever(cfg=cfg)
    lexical = ElasticsearchLexicalRetriever(index_name="hd_chunks")

    if reindex:
        print("Reindex enabled: rebuilding indexes")

        documents = ingest_all(cfg=cfg)
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        dense._store.clear_index()
        lexical.clear_index()

        hybrid = HybridRetriever(dense=dense, lexical=lexical, alpha=0.6)
        hybrid.build_index(chunks)

        print("Indexing finished")
        return hybrid

    print("Using existing indexes")
    return HybridRetriever(dense=dense, lexical=lexical, alpha=0.6)

