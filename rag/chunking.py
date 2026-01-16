"""
Provides functions for splitting documents into smaller chunks.
"""
from __future__ import annotations

import logging
from pydantic import BaseModel
from rag.ingest import RawDocument

from support_function.detect_function import detect_language

log = logging.getLogger(__name__)

class Chunk(BaseModel):
    """A Pydantic model for a text chunk from a document."""
    id: str
    doc_id: str
    text: str
    order: int
    start_char: int
    end_char: int
    doc_name: str
    language: str | None = None
    category: str | None = None
    allowed_roles: str | None = None


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Splits a single document into a list of `Chunk` objects."""
    text = doc.text or ""
    doc_name = doc.source or "unknown"
    length = len(text)

    if length == 0:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть в диапазоне [0, chunk_size)")

    chunks: list[Chunk] = []
    start = 0
    n = 0
    while start < length:
        end = min(start + chunk_size, length)

        chunk_text = text[start:end]
        language = detect_language(chunk_text)
        chunk = Chunk(
            id=f"{doc.id}::chunk_{n}",
            doc_id=doc.id,
            doc_name=doc_name,
            text=chunk_text,
            order=n,
            start_char=start,
            end_char=end,
            language=language,
            category=doc.category,
            allowed_roles=doc.allowed_roles,
        )
        chunks.append(chunk)

        if end == length:
            break

        start = end - overlap
        n += 1

    return chunks


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Splits a list of documents into a single list of `Chunk` objects."""
    log.info(f"Starting chunking for {len(docs)} documents...", extra={'log_type': 'INFO'})
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc=doc, chunk_size=chunk_size, overlap=overlap)
        log.debug(
            "Chunked document %s into %d chunks. First chunk roles: %s",
            doc.source,
            len(chunks),
            chunks[0].allowed_roles if chunks else "N/A",
        )
        all_chunks.extend(chunks)
    log.info(f"Finished chunking. Created {len(all_chunks)} chunks.", extra={'log_type': 'INFO'})
    return all_chunks
