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

    enhancement_mode = (pipeline.cfg.query_enhancement_mode or "single").lower()
    if enhancement_mode not in {"single", "multi"}:
        log.warning(
            "Unknown query_enhancement_mode='%s', fallback to 'single'.",
            pipeline.cfg.query_enhancement_mode,
            extra={"log_type": "WARNING"},
        )
        enhancement_mode = "single"

    base_queries = pipeline.decomposer.decompose(query)

    final_context_for_llm = []
    initial_contexts_for_logging = {}
    neighbors_forward = max(0, int(pipeline.cfg.neighbors_forward))
    neighbors_backward = max(0, int(pipeline.cfg.neighbors_backward))
    rerank_top_k = max(1, int(pipeline.cfg.rerank_top_k))

    total_retrieve_s = 0.0
    total_rerank_s = 0.0
    total_compress_s = 0.0
    total_enhance_s = 0.0

    for base_query in base_queries:
        if pipeline.query_enhancer:
            enhance_start = time.perf_counter()
            v, h = pipeline.query_enhancer.enhance(base_query)
            total_enhance_s += time.perf_counter() - enhance_start
        else:
            v, h = [], None

        if (pipeline.cfg.log_mode & 1) and (v or h):
            log.info(
                "Query enhancements for subquery: variations=%d, hypothetical=%s",
                len(v),
                "yes" if h else "no",
                extra={"log_type": "ENHANCEMENT"},
            )
            for i, item in enumerate(v, start=1):
                log.info("Variation %d: %s", i, item, extra={"log_type": "ENHANCEMENT"})
            if h:
                log.info("Hypothetical answer: %s", h, extra={"log_type": "ENHANCEMENT"})

        if enhancement_mode == "single" and (v or h):
            query_set = [_build_combined_query(base_query, v, h)]
        else:
            extras = []
            if v:
                extras.extend(v)
            if h:
                extras.append(h)
            query_set = _dedupe_queries([base_query] + extras)

        results = []
        retrieve_start = time.perf_counter()
        for q in query_set:
            results.extend(
                pipeline.retriever.retrieve(q, language=language, category=category)
            )
        total_retrieve_s += time.perf_counter() - retrieve_start

        if not results:
            continue

        rerank_start = time.perf_counter()
        reranked_results = pipeline.reranker.rerank(base_query, results, top_k=rerank_top_k)
        total_rerank_s += time.perf_counter() - rerank_start
        reranked_results = pipeline.acl_filter.filter(reranked_results, user_role=user_role)
        if not reranked_results:
            continue

        combined_chunks = []
        seen_chunk_ids = set()
        for r in reranked_results:
            context_chunks = pipeline.retriever._dense._store.get_neighbors_window(
                r["main_chunk"],
                neighbors_backward=neighbors_backward,
                neighbors_forward=neighbors_forward,
            )
            for c in context_chunks:
                if c.id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(c.id)
                combined_chunks.append(c)

        if pipeline.cfg.use_compressor:
            compress_start = time.perf_counter()
            compressed_context = pipeline.compressor.compress(question=base_query, chunks=combined_chunks)
            total_compress_s += time.perf_counter() - compress_start
        else:
            compressed_context = "\n\n".join([c.text for c in combined_chunks]).strip()

        item = {**reranked_results[0], "compressed_context": compressed_context}
        final_context_for_llm.append(item)

    if pipeline.cfg.log_mode & 2:
        log.info("Query enhancement time: %.2fs", total_enhance_s, extra={"log_type": "INFO"})
        log.info("Retrieval time: %.2fs", total_retrieve_s, extra={"log_type": "INFO"})
        log.info("Rerank time: %.2fs", total_rerank_s, extra={"log_type": "INFO"})
        if pipeline.cfg.use_compressor:
            log.info("Compression time: %.2fs", total_compress_s, extra={"log_type": "INFO"})

    if not final_context_for_llm:
        log.warning("No documents found by retriever.", extra={"log_type": "WARNING"})
        answer_result = AnswerResult(answer=pipeline.cfg.no_data_response)
        if return_debug:
            return answer_result, QueryDebug(initial_contexts={})
        return answer_result

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
