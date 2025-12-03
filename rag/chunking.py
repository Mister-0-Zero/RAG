from __future__ import annotations

from pydantic import BaseModel
from rag.ingest import RawDocument


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    order: int
    start_char: int
    end_char: int


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    text = doc.text or ""
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

        chunk = Chunk(
            id=f"{doc.id}::chunk_{n}",
            doc_id=doc.id,
            text=text[start:end],
            order=n,
            start_char=start,
            end_char=end,
        )
        chunks.append(chunk)

        if end == length:
            break  # дошли до конца текста

        start = end - overlap
        n += 1

    return chunks


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc=doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
