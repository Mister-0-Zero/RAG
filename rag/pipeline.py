from __future__ import annotations

from typing import Tuple

from rag.config import RAGConfig
from rag.ingest import ingest_all
from rag.chunking import chunk_documents, Chunk
from rag.retrieval import DenseRetriever

def build_dense_retriever(
    cfg: RAGConfig | None = None,
    chunk_size: int = 800,
    overlap: int = 200,
) -> tuple[DenseRetriever, list[Chunk]]:
    cfg = cfg or RAGConfig()

    documents = ingest_all(cfg=cfg)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    retriever = DenseRetriever(cfg=cfg)
    retriever.build_index(chunks)

    return retriever, chunks
