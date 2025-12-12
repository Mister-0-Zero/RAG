from __future__ import annotations

from typing import Any

from rag.chunking import Chunk
from rag.retrieval import DenseRetriever
from rag.lexical_es import ElasticsearchLexicalRetriever


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        lexical: ElasticsearchLexicalRetriever,
        alpha: float = 0.5,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._alpha = alpha

    def build_index(self, chunks: list[Chunk], clear: bool = True) -> None:
        self._dense.build_index(chunks, clear=clear)
        self._lexical.index_chunks(chunks, clear=clear)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
        category: str | None = None,
        neighbors: int = 0,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:

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

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        # 4) add neighbors only for final top
        final: list[dict[str, Any]] = []
        for r in top:
            ch: Chunk = r["main_chunk"]
            context = self._dense.get_neighbors(ch, neighbors=neighbors)

            final.append(
                {
                    "chunk": context,
                    "main_chunk": ch,
                    "score": r["score"],
                    "dense_score": r["dense_score"],
                    "lexical_score": r["lexical_score"],
                    "lexical_norm": r["lexical_norm"],
                    "metadata": r["metadata"],
                }
            )

        return final
