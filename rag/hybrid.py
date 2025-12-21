"""
Provides a hybrid retriever that combines dense and lexical search results.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.chunking import Chunk
from rag.retrieval import DenseRetriever
from rag.lexical_es import ElasticsearchLexicalRetriever

log = logging.getLogger(__name__)


class HybridRetriever:
    """Combines results from a DenseRetriever and an ElasticsearchLexicalRetriever."""
    def __init__(
        self,
        dense: DenseRetriever,
        lexical: ElasticsearchLexicalRetriever,
        alpha: float = 0.5,
    ) -> None:
        """
        Initializes the HybridRetriever with dense and lexical retrievers and a weighting factor.
        """
        self._dense = dense
        self._lexical = lexical
        self._alpha = alpha
        log.info(f"HybridRetriever initialized with alpha={alpha}", extra={'log_type': 'INFO'})

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        """Builds the index for both the dense and lexical retrievers."""
        self._dense.build_index(chunks, clear=clear)
        self._lexical.index_chunks(chunks, clear=clear)

    def retrieve(
        self,
        query: str,
        language: str | None = None,
        category: str | None = None,
        candidate_k: int = 24,
    ) -> list[dict[str, Any]]:
        """Retrieves and fuses results from dense and lexical retrievers for a given query."""
        dense_hits = self._dense.retrieve(
            query,
            top_k=candidate_k,
            language=language,
            category=category,
            neighbors=0,
        )

        lex_hits = self._lexical.search(
            query,
            top_k=candidate_k,
            language=language,
            category=category,
        )
        log.info(f"Retrieved {len(dense_hits)} dense hits and {len(lex_hits)} lexical hits.", extra={'log_type': 'INFO'})

        candidates: dict[str, dict[str, Any]] = {}

        # 1) merge dense
        for h in dense_hits:
            ch: Chunk = h["main_chunk"]
            dense_score = h.get("score")
            if dense_score is None:
                continue

            cid = ch.id
            entry = candidates.setdefault(
                cid,
                {"chunk": ch, "dense": 0.0, "lex": 0.0, "dense_meta": {}},
            )
            entry["dense"] = max(entry["dense"], float(dense_score))
            entry["dense_meta"] = h.get("metadata", {}) or {}

        # 2) merge lexical
        for h in lex_hits:
            ch: Chunk = h["chunk"]
            lex_score = float(h["score"])

            cid = ch.id
            entry = candidates.setdefault(
                cid,
                {"chunk": ch, "dense": 0.0, "lex": 0.0, "dense_meta": {}},
            )
            entry["lex"] = max(entry["lex"], lex_score)

        # 3) normalize lexical (divide by max)
        lex_vals = [v["lex"] for v in candidates.values() if v["lex"] > 0]
        max_lex = max(lex_vals) if lex_vals else 1.0

        scored: list[dict[str, Any]] = []
        for v in candidates.values():
            dense_s = float(v["dense"])
            lex_s = float(v["lex"])
            lex_norm = (lex_s / max_lex) if max_lex > 0 else 0.0

            hybrid_score = self._alpha * dense_s + (1.0 - self._alpha) * lex_norm

            scored.append(
                {
                    "main_chunk": v["chunk"],
                    "score": hybrid_score,
                    "dense_score": dense_s,
                    "lexical_score": lex_s,
                    "lexical_norm": lex_norm,
                    "metadata": v["dense_meta"],
                }
            )

        return scored
