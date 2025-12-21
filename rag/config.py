from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import torch

class RAGConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    embeddings_model: str = "intfloat/multilingual-e5-base"
    chroma_db: Path = Path("data/vector_db")
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None

    def model_post_init(self, __context):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
