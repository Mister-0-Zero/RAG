from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from rag.config import RAGConfig


class EmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cfg: RAGConfig | None = None,
    ) -> None:
        self._cfg = cfg or RAGConfig()
        self.model_name = model_name or self._cfg.embeddings_model

        if device is not None:
            dev = device.strip().lower()
            if dev not in {"cpu", "cuda"}:
                raise ValueError("device must be 'cpu' or 'cuda'")
            self.device = dev
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Эмбеддинги для списка текстов."""
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг для одного текста (запроса)."""
        return self.embed_texts([text])[0]
