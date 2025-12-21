"""
Provides a reranker for search results using a CrossEncoder model.
"""
import logging
from typing import Any, List
from sentence_transformers import CrossEncoder
from rag.config import RAGConfig

log = logging.getLogger(__name__)

class Reranker:
    """Uses a CrossEncoder model to rerank a list of candidate documents."""
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """Initializes the Reranker, loading the specified CrossEncoder model."""
        cfg = RAGConfig()
        self.model_name = model_name or cfg.rerank_model
        log.info(f"Loading reranker model: {self.model_name}", extra={'log_type': 'INFO'})
        self.model = CrossEncoder(self.model_name, device=cfg.device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[dict[str, Any]],
        top_k: int = 3,
    ) -> List[dict[str, Any]]:
        """Reranks a list of candidate documents against a query and returns the top_k results."""
        if not candidates:
            return []
        
        log.info(f"Reranking {len(candidates)} candidates...", extra={'log_type': 'INFO'})

        pairs = [
            (query, c["main_chunk"].text)
            for c in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return candidates[:top_k]
