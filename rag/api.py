"""
Public API for building and running the RAG pipeline as a module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any

from rag.config import RAGConfig
from rag.compressor import ContextCompressor
from rag.pipeline import build_hybrid_retriever
from rag.rerank import Reranker
from search.es_client import get_es, check_es_or_die
from support_function.detect_function import detect_language, detect_category
from rag.llm import init_llm_client, init_query_llm_client
from rag.query_enhancer import QueryEnhancer
from rag.answer import AnswerGenerator, AnswerResult
from rag.query_decomposer import QueryDecomposer
from rag.acl_runtime import ACLRuntimeFilter

log = logging.getLogger(__name__)


@dataclass
class QueryDebug:
    initial_contexts: dict[str, str]


@dataclass
class RAGPipeline:
    retriever: Any
    reranker: Any
    compressor: Any
    answer_generator: AnswerGenerator
    decomposer: QueryDecomposer
    cfg: RAGConfig
    acl_filter: ACLRuntimeFilter
    query_enhancer: QueryEnhancer | None

    def query(
        self,
        query: str,
        user_role: str = "guest",
        return_debug: bool = False,
    ) -> AnswerResult | tuple[AnswerResult, QueryDebug]:
        return process_query(query, self, user_role=user_role, return_debug=return_debug)


def build_pipeline(
    reindex: bool = False,
    cfg: Optional[RAGConfig] = None,
    chunk_size: int = 600,
    overlap: int = 150,
) -> RAGPipeline:
    """
    Initializes and wires together the components of the RAG pipeline.
    """
    cfg = cfg or RAGConfig()
    log.info("Initializing RAG pipeline...", extra={"log_type": "INFO"})

    es = get_es()
    check_es_or_die(es)

    llm_client = init_llm_client(cfg)
    decomposer = QueryDecomposer(llm_client, cfg)
    answer_generator = AnswerGenerator(llm_client, cfg)
    acl_filter = ACLRuntimeFilter(cfg)

    query_enhancer = None
    if cfg.query_variations_count > 0 or cfg.query_use_hypothetical_answer:
        query_llm_client = init_query_llm_client(cfg)
        query_enhancer = QueryEnhancer(query_llm_client, cfg)

    log.info("Building hybrid retriever...", extra={"log_type": "INFO"})
    retriever = build_hybrid_retriever(cfg=cfg, chunk_size=chunk_size, overlap=overlap, reindex=reindex)
    log.info("Retriever is ready.", extra={"log_type": "INFO"})

    reranker = Reranker()
    compressor = ContextCompressor(cfg)

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        compressor=compressor,
        answer_generator=answer_generator,
        decomposer=decomposer,
        cfg=cfg,
        acl_filter=acl_filter,
        query_enhancer=query_enhancer,
    )


def process_query(
    query: str,
    pipeline: RAGPipeline,
    user_role: str,
    answer_mode: bool = True,
    return_debug: bool = False,
) -> AnswerResult | tuple[AnswerResult, QueryDebug]:
    """
    Processes a single user query through the RAG pipeline.
    Returns an AnswerResult, and optionally initial contexts for debugging/logging.
    """
    language = detect_language(query)
    category = detect_category(query)
    log.info("Language: %s, Category: %s", language, category, extra={"log_type": "METADATA"})

    enhance_start = time.perf_counter()
    if pipeline.query_enhancer:
        variations, hypo = pipeline.query_enhancer.enhance(query)
    else:
        variations, hypo = [], None
    enhance_s = time.perf_counter() - enhance_start

    if pipeline.cfg.log_mode & 2:
        log.info("Query enhancement time: %.2fs", enhance_s, extra={"log_type": "INFO"})

    if (pipeline.cfg.log_mode & 1) and (variations or hypo):
        log.info(
            "Query enhancements: variations=%d, hypothetical=%s",
            len(variations),
            "yes" if hypo else "no",
            extra={"log_type": "ENHANCEMENT"},
        )
        for i, v in enumerate(variations, start=1):
            log.info("Variation %d: %s", i, v, extra={"log_type": "ENHANCEMENT"})
        if hypo:
            log.info("Hypothetical answer: %s", hypo, extra={"log_type": "ENHANCEMENT"})

    enhancement_mode = (pipeline.cfg.query_enhancement_mode or "single").lower()
    if enhancement_mode not in {"single", "multi"}:
        log.warning(
            "Unknown query_enhancement_mode='%s', fallback to 'single'.",
            pipeline.cfg.query_enhancement_mode,
            extra={"log_type": "WARNING"},
        )
        enhancement_mode = "single"

    if enhancement_mode == "single" and (variations or hypo):
        combined_query = _build_combined_query(query, variations, hypo)
        subqueries = [combined_query]
    else:
        base_queries = pipeline.decomposer.decompose(query)
        extra_queries = []
        if variations:
            extra_queries.extend(variations)
        if hypo:
            extra_queries.append(hypo)

        subqueries = _dedupe_queries(base_queries + extra_queries)

    results = []
    retrieve_start = time.perf_counter()
    for sq in subqueries:
        results.extend(
            pipeline.retriever.retrieve(sq, language=language, category=category)
        )
    retrieve_s = time.perf_counter() - retrieve_start

    if pipeline.cfg.log_mode & 2:
        log.info("Retrieval time: %.2fs", retrieve_s, extra={"log_type": "INFO"})

    if not results:
        log.warning("No documents found by retriever.", extra={"log_type": "WARNING"})
        answer_result = AnswerResult(answer=pipeline.cfg.no_data_response)
        if return_debug:
            return answer_result, QueryDebug(initial_contexts={})
        return answer_result

    rerank_start = time.perf_counter()
    reranked_results = pipeline.reranker.rerank(query, results, top_k=2)
    rerank_s = time.perf_counter() - rerank_start

    if pipeline.cfg.log_mode & 2:
        log.info("Rerank time: %.2fs", rerank_s, extra={"log_type": "INFO"})
    reranked_results = pipeline.acl_filter.filter(reranked_results, user_role=user_role)
    if not reranked_results:
        log.warning(
            "All data was not found or was filtered out, with the role=%s",
            user_role,
            extra={"log_type": "WARNING"},
        )
        answer_result = AnswerResult(answer=pipeline.cfg.no_data_response)
        if return_debug:
            return answer_result, QueryDebug(initial_contexts={})
        return answer_result

    final_context_for_llm = []
    initial_contexts_for_logging = {}

    compress_start = time.perf_counter()
    for r in reranked_results:
        neighbors_forward = max(0, int(pipeline.cfg.neighbors_forward))
        neighbors_backward = max(0, int(pipeline.cfg.neighbors_backward))
        context_chunks = pipeline.retriever._dense._store.get_neighbors_window(
            r["main_chunk"],
            neighbors_backward=neighbors_backward,
            neighbors_forward=neighbors_forward,
        )

        if pipeline.cfg.log_mode & 4:
            context_text = " ".join([c.text for c in context_chunks])
            initial_contexts_for_logging[r["main_chunk"].id] = context_text
            log.info(
                "Context for compression (%s): %s",
                r["main_chunk"].doc_name,
                context_text,
                extra={"log_type": "CONTEXT_AFTER_RERANK"},
            )

        compressed_context = pipeline.compressor.compress(question=query, chunks=context_chunks)
        final_context_for_llm.append({**r, "compressed_context": compressed_context})
    compress_s = time.perf_counter() - compress_start

    if pipeline.cfg.log_mode & 2:
        log.info("Compression time: %.2fs", compress_s, extra={"log_type": "INFO"})

    if not answer_mode:
        context_text = _build_context_text(final_context_for_llm)
        answer_result = AnswerResult(
            answer=context_text or pipeline.cfg.no_data_response,
            citations=None,
            prompt=None,
            final_context=final_context_for_llm,
        )
    else:
        answer_result = pipeline.answer_generator.generate(query=query, final=final_context_for_llm)

    if return_debug:
        return answer_result, QueryDebug(initial_contexts=initial_contexts_for_logging)
    return answer_result


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen = set()
    result = []
    for q in queries:
        if not isinstance(q, str):
            continue
        text = q.strip()
        if len(text) < 3:
            continue
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _build_combined_query(query: str, variations: list[str], hypo: str | None) -> str:
    parts = ["ORIGINAL QUERY:", query]

    if variations:
        parts.append("QUERY VARIATIONS:")
        for i, v in enumerate(variations, start=1):
            parts.append(f"VARIATION {i}: {v}")

    if hypo:
        parts.append("HYPOTHETICAL ANSWER:")
        parts.append(hypo)

    return "\n".join(parts)


def _build_context_text(final: list[dict]) -> str:
    parts = []
    for item in final:
        text = item.get("compressed_context", "")
        if text and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def log_final_result(
    answer_result: AnswerResult,
    initial_contexts: dict[str, str],
    cfg: RAGConfig,
) -> None:
    """Logs the final answer from the LLM."""
    log.info(answer_result.answer, extra={"log_type": "MODEL_RESPONSE"})

    if answer_result.citations and (cfg.log_mode & 2):
        log.info("Sources: %s", ", ".join(answer_result.citations), extra={"log_type": "INFO"})
