from __future__ import annotations

from pydantic import BaseModel
from rag.ingest import RawDocument


def detect_language(text: str) -> str | None:
    cyrillic_chars = sum(1 for char in text if 'а' <= char <= 'я' or 'А' <= char <= 'Я')
    latin_chars = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z')

    if cyrillic_chars == 0 and latin_chars == 0:
        return "mixed"

    relationship = min(cyrillic_chars, latin_chars) / max(cyrillic_chars, latin_chars)

    if relationship > 0.6:
        return "mixed"
    elif cyrillic_chars > latin_chars:
        return "ru"
    elif latin_chars > cyrillic_chars:
        return "en"
    else:
        return "mixed"

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    order: int
    start_char: int
    end_char: int
    language: str | None = None
    category: str | None = None


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    print(f"Начало чанкинга документа: {doc.id}, категория: {doc.category}")
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

        chunk_text = text[start:end]
        language = detect_language(chunk_text)
        chunk = Chunk(
            id=f"{doc.id}::chunk_{n}",
            doc_id=doc.id,
            text=chunk_text,
            order=n,
            start_char=start,
            end_char=end,
            language=language,
            category=doc.category,
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
