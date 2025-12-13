from __future__ import annotations

from typing import Tuple

from rag.config import RAGConfig
from rag.ingest import ingest_all
from rag.chunking import chunk_documents, Chunk
from rag.hybrid import HybridRetriever
from rag.retrieval import DenseRetriever
from rag.lexical_es import ElasticsearchLexicalRetriever

def build_hybrid_retriever(
    cfg: RAGConfig | None = None,
    chunk_size: int = 800,
    overlap: int = 200,
) -> tuple[HybridRetriever, list[Chunk]]:
    cfg = cfg or RAGConfig()


    documents = ingest_all(cfg=cfg)
    print()

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    print()

    dense = DenseRetriever(cfg=cfg)
    lexical = ElasticsearchLexicalRetriever(index_name="hd_chunks")
    retriever = HybridRetriever(dense=dense, lexical=lexical, alpha=0.6)

    retriever.build_index(chunks)
    print()

    return retriever, chunks
