from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path

class RAGConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_db: Path = Path("data/vector_db")