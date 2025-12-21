from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import torch

class RAGConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    embeddings_model: str = "intfloat/multilingual-e5-base"
    chroma_db: Path = Path("data/vector_db")
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_name_for_compressor: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_tokens_after_compressed_per_result_: int = 256
    temperature_model_compressor: float = 0.0
    device: str | None = None

    def model_post_init(self, __context):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
