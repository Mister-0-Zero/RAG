from typing import Any, List
from sentence_transformers import CrossEncoder
from rag.chunking import Chunk
from rag.config import RAGConfig

class Reranker:
    def __init__(
        self,
        model_name: str = RAGConfig().rerank_model,
        batch_size: int = 16,
    ) -> None:
        self.model = CrossEncoder(model_name, device=RAGConfig().device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[dict[str, Any]],
        top_k: int = 3,
    ) -> List[dict[str, Any]]:

        if not candidates:
            return []

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
