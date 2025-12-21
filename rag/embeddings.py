"""
Provides an interface to a sentence-transformer embedding model.
"""
from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

from rag.config import RAGConfig

log = logging.getLogger(__name__)


class EmbeddingModel:
    """Manages the loading and usage of a sentence-transformer model for creating embeddings."""
    def __init__(
        self,
        model_name: str | None = None,
        cfg: RAGConfig | None = None,
    ) -> None:
        """Initializes the EmbeddingModel, loading the specified model and setting the device."""
        self._cfg = cfg or RAGConfig()
        self.model_name = model_name or self._cfg.embeddings_model
        log.info(f"Loading embedding model: {self.model_name}", extra={'log_type': 'INFO'})

        self.device = self._cfg.device
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Creates embeddings for a list of texts."""
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Creates an embedding for a single query text."""
        return self.embed_texts([text])[0]
