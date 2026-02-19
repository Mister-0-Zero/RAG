"""
Provides functions for splitting documents into smaller chunks.
"""
from __future__ import annotations

import logging
import re
from pydantic import BaseModel

from rag.ingest import RawDocument
from rag.config import RAGConfig
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
    section_title: str | None = None
    language: str | None = None
    category: str | None = None
    allowed_roles: str | None = None


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = 800,
    overlap: int = 200,
    cfg: RAGConfig | None = None,
) -> list[Chunk]:
    """Splits a list of documents into a single list of `Chunk` objects."""
    log.info(f"Starting chunking for {len(docs)} documents...", extra={'log_type': 'INFO'})
    all_chunks: list[Chunk] = []

    chunking_mode = getattr(cfg, "chunking_mode", "size") if cfg else "size"
    delimiters = getattr(cfg, "chunk_delimiters", None) if cfg else None
    delimiters_are_regex = getattr(cfg, "chunk_delimiters_are_regex", False) if cfg else False
    include_delimiter = getattr(cfg, "chunk_delimiter_included", False) if cfg else False

    for doc in docs:
        if chunking_mode == "delimiter":
            chunks = chunk_by_delimiters(
                doc=doc,
                delimiters=delimiters or [],
                delimiters_are_regex=delimiters_are_regex,
                include_delimiter=include_delimiter,
            )
        elif chunking_mode == "hybrid":
            chunks = chunk_by_delimiters_and_size(
                doc=doc,
                delimiters=delimiters or [],
                delimiters_are_regex=delimiters_are_regex,
                include_delimiter=include_delimiter,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        else:
            chunks = chunk_by_size(
                doc=doc,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        log.debug(
            "Chunked document %s into %d chunks. First chunk roles: %s",
            doc.source,
            len(chunks),
            chunks[0].allowed_roles if chunks else "N/A",
        )
        all_chunks.extend(chunks)

    log.info(f"Finished chunking. Created {len(all_chunks)} chunks.", extra={'log_type': 'INFO'})
    return all_chunks


def chunk_by_size(
    doc: RawDocument,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Splits a single document into chunks by size/overlap."""
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
        chunks.append(_make_chunk(doc, doc_name, text[start:end], n, start, end))

        if end == length:
            break

        start = end - overlap
        n += 1

    return chunks


def chunk_by_delimiters(
    doc: RawDocument,
    delimiters: list[str],
    delimiters_are_regex: bool = False,
    include_delimiter: bool = False,
) -> list[Chunk]:
    """Splits a single document into chunks using delimiters or regex patterns."""
    text = doc.text or ""
    doc_name = doc.source or "unknown"
    length = len(text)

    if length == 0:
        return []

    if not delimiters:
        return _fallback_full_text_chunk(doc, doc_name, text, length)

    pattern = _build_delimiter_pattern(delimiters, delimiters_are_regex)
    matches = list(re.finditer(pattern, text))

    if not matches:
        return _fallback_full_text_chunk(doc, doc_name, text, length)

    ranges = _ranges_from_matches(matches, length, include_delimiter)
    return _chunks_from_ranges(doc, doc_name, text, ranges)


def chunk_by_delimiters_and_size(
    doc: RawDocument,
    delimiters: list[str],
    delimiters_are_regex: bool = False,
    include_delimiter: bool = False,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Splits by delimiters, then applies size chunking inside large blocks."""
    text = doc.text or ""
    doc_name = doc.source or "unknown"
    length = len(text)

    if length == 0:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть в диапазоне [0, chunk_size)")

    if not delimiters:
        return _fallback_full_text_chunk(doc, doc_name, text, length)

    pattern = _build_delimiter_pattern(delimiters, delimiters_are_regex)
    matches = list(re.finditer(pattern, text))

    if not matches:
        return _fallback_full_text_chunk(doc, doc_name, text, length)

    ranges = _ranges_from_matches(matches, length, include_delimiter)
    chunks: list[Chunk] = []
    n = 0
    for start, end in ranges:
        block_text = text[start:end].strip()
        if not block_text:
            continue

        block_len = len(block_text)
        if block_len <= chunk_size:
            chunks.append(_make_chunk(doc, doc_name, block_text, n, start, end))
            n += 1
            continue

        sub_start = start
        while sub_start < end:
            sub_end = min(sub_start + chunk_size, end)
            sub_text = text[sub_start:sub_end].strip()
            if sub_text:
                chunks.append(_make_chunk(doc, doc_name, sub_text, n, sub_start, sub_end))
                n += 1
            if sub_end == end:
                break
            sub_start = sub_end - overlap

    return chunks


def _build_delimiter_pattern(delimiters: list[str], delimiters_are_regex: bool) -> str:
    if delimiters_are_regex:
        return "|".join(f"(?:{d})" for d in delimiters)
    return "|".join(re.escape(d) for d in delimiters)


def _ranges_from_matches(matches: list[re.Match], length: int, include_delimiter: bool) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for i, match in enumerate(matches):
        start = match.start() if include_delimiter else match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else length
        if start < end:
            ranges.append((start, end))
    return ranges


def _chunks_from_ranges(
    doc: RawDocument,
    doc_name: str,
    text: str,
    ranges: list[tuple[int, int]],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for start, end in ranges:
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        chunks.append(_make_chunk(doc, doc_name, chunk_text, n, start, end))
        n += 1
    return chunks


def _fallback_full_text_chunk(
    doc: RawDocument,
    doc_name: str,
    text: str,
    length: int,
) -> list[Chunk]:
    log.warning(
        "Delimiter not found in document %s. Using full text as one chunk (size=%d).",
        doc_name,
        length,
        extra={'log_type': 'WARNING'},
    )
    return [_make_chunk(doc, doc_name, text, 0, 0, length)]


def _make_chunk(
    doc: RawDocument,
    doc_name: str,
    chunk_text: str,
    order: int,
    start: int,
    end: int,
) -> Chunk:
    language = detect_language(chunk_text)
    section_title = _extract_section_title(chunk_text)
    return Chunk(
        id=f"{doc.id}::chunk_{order}",
        doc_id=doc.id,
        doc_name=doc_name,
        text=chunk_text,
        order=order,
        start_char=start,
        end_char=end,
        language=language,
        category=doc.category,
        allowed_roles=doc.allowed_roles,
        section_title=section_title,
    )


def _extract_section_title(text: str) -> str | None:
    if not text:
        return None
    first_line = text.strip().splitlines()[0].strip()
    if not first_line:
        return None

    quoted = re.search(r'"([^"]+)"', first_line)
    if quoted:
        return quoted.group(1).strip()

    if first_line.lower().startswith("section:"):
        title = first_line.split(":", 1)[1].strip()
        return title.strip('"').strip() or None

    if len(first_line) <= 120:
        return first_line

    return None
