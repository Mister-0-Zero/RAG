from __future__ import annotations

from pathlib import Path

from docx import Document
import fitz
from pydantic import BaseModel

from rag.config import RAGConfig


class RawDocument(BaseModel):
    """Сырой документ до чанкинга."""

    id: str
    path: Path
    text: str
    source: str
    doc_type: str
    language: str | None = None


def normalize_text(text: str) -> str:
    """Простая нормализация текста."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in text.split("\n")]

    normalized_lines: list[str] = []

    for line in lines:
        if line != "":
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def read_txt(path: Path) -> str:
    with path.open(encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(path: Path) -> str:
    doc = Document(path)
    texts = [p.text for p in doc.paragraphs]
    return "\n".join(texts)


def read_pdf(path: Path) -> str:
    texts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            texts.append(page.get_text())
    return "\n".join(texts)


def ingest_directory(path_dir: Path) -> list[RawDocument]:
    """Читает txt/docx/pdf из директории и возвращает список RawDocument."""
    documents: list[RawDocument] = []

    for file in path_dir.rglob("*"):
        if not file.is_file():
            continue

        suffix = file.suffix.lower()

        if suffix == ".txt":
            raw_text = read_txt(file)
            doc_type = "txt"
        elif suffix == ".docx":
            raw_text = read_docx(file)
            doc_type = "docx"
        elif suffix == ".pdf":
            raw_text = read_pdf(file)
            doc_type = "pdf"
        else:
            # временно просто предупреждаем и пропускаем
            print(
                f"Тип файла {suffix} не поддерживается. "
                "Поддерживаемые типы: txt, docx, pdf."
            )
            continue

        text = normalize_text(raw_text)

        documents.append(
            RawDocument(
                id=file.stem,
                path=file,
                text=text,
                source=file.name,
                doc_type=doc_type,
            )
        )

    return documents


def ingest_all(cfg: RAGConfig | None = None) -> list[RawDocument]:
    cfg = cfg or RAGConfig()
    return ingest_directory(cfg.data_raw)
