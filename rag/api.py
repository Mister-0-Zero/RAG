"""
Public API for building and running the RAG pipeline as a module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from rag.config import RAGConfig
from rag.compressor import ContextCompressor
from rag.pipeline import build_hybrid_retriever
from rag.rerank import Reranker
from search.es_client import get_es, check_es_or_die
from support_function.detect_function import detect_language, detect_category
from rag.llm import init_llm_client
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
    neighbors: int
    acl_filter: ACLRuntimeFilter

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
    neighbors: int = 3,
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
        neighbors=neighbors,
        acl_filter=acl_filter,
    )


def process_query(
    query: str,
    pipeline: RAGPipeline,
    user_role: str,
    return_debug: bool = False,
) -> AnswerResult | tuple[AnswerResult, QueryDebug]:
    """
    Processes a single user query through the RAG pipeline.
    Returns an AnswerResult, and optionally initial contexts for debugging/logging.
    """
    language = detect_language(query)
    category = detect_category(query)
    log.info("Language: %s, Category: %s", language, category, extra={"log_type": "METADATA"})

    subqueries = pipeline.decomposer.decompose(query)

    results = []
    for sq in subqueries:
        results.extend(
            pipeline.retriever.retrieve(sq, language=language, category=category)
        )

    if not results:
        log.warning("No documents found by retriever.", extra={"log_type": "WARNING"})
        answer_result = AnswerResult(answer=pipeline.cfg.no_data_response)
        if return_debug:
            return answer_result, QueryDebug(initial_contexts={})
        return answer_result

    reranked_results = pipeline.reranker.rerank(query, results, top_k=3)
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

    for r in reranked_results:
        context_chunks = pipeline.retriever._dense._store.get_neighbors(
            r["main_chunk"], neighbors_forward=pipeline.neighbors
        )

        if pipeline.cfg.extended_logs:
            initial_contexts_for_logging[r["main_chunk"].id] = " ".join([c.text for c in context_chunks])

        compressed_context = pipeline.compressor.compress(question=query, chunks=context_chunks)
        final_context_for_llm.append({**r, "compressed_context": compressed_context})

    answer_result = pipeline.answer_generator.generate(query=query, final=final_context_for_llm)

    if return_debug:
        return answer_result, QueryDebug(initial_contexts=initial_contexts_for_logging)
    return answer_result


def log_final_result(
    answer_result: AnswerResult,
    initial_contexts: dict[str, str],
    cfg: RAGConfig,
) -> None:
    """
    Logs the final answer from the LLM. If extended_logs is enabled in the
    configuration, it also logs detailed context, scores, and the prompt.
    """
    if not cfg.extended_logs:
        log.info(answer_result.answer, extra={"log_type": "MODEL_RESPONSE"})
        return

    log.info("=" * 20 + " FINAL ANSWER " + "=" * 20, extra={"log_type": "MODEL_RESPONSE"})
    log.info(answer_result.answer, extra={"log_type": "MODEL_RESPONSE"})

    if answer_result.citations:
        log.info("Sources: %s", ", ".join(answer_result.citations), extra={"log_type": "METADATA"})

    log.info("--- EXTENDED LOGS ---", extra={"log_type": "INFO"})

    if answer_result.prompt:
        log.info("LLM Prompt:\n%s", answer_result.prompt, extra={"log_type": "DEBUG"})

    if answer_result.final_context:
        for item in answer_result.final_context:
            main_chunk = item.get("main_chunk")
            if not main_chunk:
                continue

            log.info("-" * 15, extra={"log_type": "DEBUG"})
            score = item.get("score", 0.0)
            rerank_score = item.get("rerank_score", 0.0)
            doc_name = main_chunk.doc_name
            log.info(
                "Source: %s (Score: %.4f, Rerank Score: %.4f)",
                doc_name,
                score,
                rerank_score,
                extra={"log_type": "DEBUG"},
            )

            initial_context = initial_contexts.get(main_chunk.id)
            if initial_context:
                log.info("Initial Context: %s", initial_context, extra={"log_type": "DEBUG"})

            compressed_context = item.get("compressed_context", "")
            log.info("Compressed Context: %s", compressed_context, extra={"log_type": "DEBUG"})
    log.info("=" * 56, extra={"log_type": "INFO"})
