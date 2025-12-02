# rag/ingest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import docx
import re

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")

@dataclass
@dataclass
class DocumentMeta:
    path: Path
    rel_path: str
    doc_type: str
    role: str       # "user" / "admin" / "?"
    source: str     # человекочитаемый источник (по умолчанию rel_path)
    lang: str       # "ru" / "en" / "mixed" / "unknown"
    category: str   # "gates" / "channels" / "centers" / "types" / "crosses" / "other"

def infer_role_and_category(rel_path: str) -> tuple[str, str]:
    filename = rel_path.split("/")[-1]

    # словарь ролей по имени файла
    role_by_name = {
        "Algorithm_for_calculating_the_Incarnation_cross.docx": "user",
        "Алгоритм_Расчёта_Инкарнационного_Креста.docx": "user",
        "Ворота_Gate.docx": "user",
        "Каналы_Chenels.txt": "user",
        "РасширеннаяИнформацияПоЦентрам_Centers.docx": "user",
        "Типы_Профили_Стратегии_Авторитеты.docx": "user",
        "Центры_Centers_КрестИнкарнации_CrossIncarnation.pdf": "user",
        "тайна.docx": "admin",
    }

    role = role_by_name.get(filename, "?")

    lower = filename.lower()
    if "ворота" in lower or "gate" in lower:
        category = "gates"
    elif "канал" in lower or "chenel" in lower or "channel" in lower:
        category = "channels"
    elif "центр" in lower or "center" in lower:
        category = "centers"
    elif "тип" in lower or "profile" in lower:
        category = "types"
    elif "крест" in lower or "cross" in lower:
        category = "crosses"
    else:
        category = "other"

    return role, category

def infer_lang(text: str) -> str:
    has_cyr = bool(CYRILLIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))

    if has_cyr and has_lat:
        return "mixed"
    if has_cyr:
        return "ru"
    if has_lat:
        return "en"
    return "unknown"

def infer_lang_from_filename(rel_path: str) -> str:
    return infer_lang(rel_path)

def scan_raw_dir(root: Path | str = Path(".data/raw")) -> List[DocumentMeta]:
    root_path = Path(root)
    docs: List[DocumentMeta] = []

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        rel_path = str(path.relative_to(root_path))
        doc_type = ext.lstrip(".")

        role, category = infer_role_and_category(rel_path)
        lang = infer_lang_from_filename(rel_path)

        docs.append(
            DocumentMeta(
                path=path,
                rel_path=rel_path,
                doc_type=doc_type,
                role=role,
                source=rel_path,
                lang=lang,
                category=category,
            )
        )

    return docs

def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except:
        return path.read_text(encoding="cp1251", errors="ignore")

def read_docx(path: Path) -> str:
    doc = docx.Document(path)
    texts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            texts.append(text)
    return "\n".join(texts)

def read_pdf(path: Path) -> str:
    return ""

def read_document(meta: DocumentMeta) -> str:
    if meta.doc_type == "txt":
        return read_txt(meta.path)
    if meta.doc_type == "docx":
        return read_docx(meta.path)
    if meta.doc_type == "pdf":
        return read_pdf(meta.path)
    return ""

if __name__ == "__main__":
    docs = scan_raw_dir()
    for d in docs:
        text = read_document(d)
        print(f"{d.rel_path} → {len(text)} chars")
