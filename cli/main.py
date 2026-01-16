"""
This module provides a command-line interface for interacting with the RAG pipeline.
It handles argument parsing, sets up logging, initializes the RAG components,
and processes user queries in a loop.
"""
from __future__ import annotations

import sys
import logging
import argparse
from types import SimpleNamespace
from dotenv import load_dotenv

from rag.config import RAGConfig
from rag.compressor import ContextCompressor
from rag.logger import setup_logging
from rag.pipeline import build_hybrid_retriever
from rag.rerank import Reranker
from search.es_client import get_es, check_es_or_die
from support_function.detect_function import detect_language, detect_category
from rag.llm import init_llm_client
from rag.answer import AnswerGenerator, AnswerResult
from rag.query_decomposer import QueryDecomposer
from rag.users import USERS
from rag.acl_runtime import ACLRuntimeFilter
from rag.auth import authenticate_user

log = logging.getLogger(__name__)

def setup_pipeline(reindex: bool, cfg: RAGConfig) -> SimpleNamespace:
    """
    Initializes and wires together the components of the RAG pipeline.

    Args:
        reindex: If True, rebuilds the search index.
        cfg: The application configuration.

    Returns:
        A namespace object containing the initialized pipeline components.
    """
    log.info("Initializing RAG pipeline...", extra={'log_type': 'INFO'})

    es = get_es()
    check_es_or_die(es)

    llm_client = init_llm_client(cfg)
    decomposer = QueryDecomposer(llm_client, cfg)
    answer_generator = AnswerGenerator(llm_client, cfg)
    acl_filter = ACLRuntimeFilter(cfg)

    log.info("Building hybrid retriever...", extra={'log_type': 'INFO'})
    retriever = build_hybrid_retriever(cfg=cfg, chunk_size=600, overlap=150, reindex=reindex)
    log.info("Retriever is ready.", extra={'log_type': 'INFO'})

    reranker = Reranker()
    compressor = ContextCompressor(cfg)

    return SimpleNamespace(
        retriever=retriever,
        reranker=reranker,
        compressor=compressor,
        answer_generator=answer_generator,
        decomposer=decomposer,
        cfg=cfg,
        neighbors=3,
        acl_filter=acl_filter
    )

def process_query(query: str, pipeline: SimpleNamespace, user_role: str) -> None:
    """
    Processes a single user query through the RAG pipeline, gathers all necessary
    information, and then logs it according to the configuration.

    Args:
        query: The user's input query.
        pipeline: The namespace object with all pipeline components.
    """

    language = detect_language(query)
    category = detect_category(query)
    log.info(f"Language: {language}, Category: {category}", extra={'log_type': 'METADATA'})

    subqueries = pipeline.decomposer.decompose(query)

    results = []
    for sq in subqueries:
        results.extend(
            pipeline.retriever.retrieve(sq, language=language, category=category)
        )
    if not results:
        log.warning("No documents found by retriever.", extra={'log_type': 'WARNING'})
        log.info(pipeline.cfg.no_data_response, extra={'log_type': 'MODEL_RESPONSE'})
        return

    reranked_results = pipeline.reranker.rerank(query, results, top_k=3)
    reranked_results = pipeline.acl_filter.filter(reranked_results, user_role=user_role)
    if not reranked_results:
        log.warning(
            "All data was not found or was filtered out, with the role=%s",
            user_role,
            extra={'log_type': 'WARNING'}
        )
        log.info(
            pipeline.cfg.no_data_response,
            extra={'log_type': 'MODEL_RESPONSE'}
        )
        return

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

    log_final_result(answer_result, initial_contexts_for_logging, pipeline.cfg)


def log_final_result(
    answer_result: AnswerResult,
    initial_contexts: dict[str, str],
    cfg: RAGConfig
) -> None:
    """
    Logs the final answer from the LLM. If extended_logs is enabled in the
    configuration, it also logs detailed context, scores, and the prompt.
    """
    if not cfg.extended_logs:
        log.info(answer_result.answer, extra={'log_type': 'MODEL_RESPONSE'})
        return

    log.info("=" * 20 + " FINAL ANSWER " + "=" * 20, extra={'log_type': 'MODEL_RESPONSE'})
    log.info(answer_result.answer, extra={'log_type': 'MODEL_RESPONSE'})

    if answer_result.citations:
        log.info(f"Sources: {', '.join(answer_result.citations)}", extra={'log_type': 'METADATA'})

    log.info("--- EXTENDED LOGS ---", extra={'log_type': 'INFO'})

    if answer_result.prompt:
        log.info(f"LLM Prompt:\n{answer_result.prompt}", extra={'log_type': 'DEBUG'})

    if answer_result.final_context:
        for item in answer_result.final_context:
            main_chunk = item.get("main_chunk")
            if not main_chunk:
                continue

            log.info("-" * 15, extra={'log_type': 'DEBUG'})
            score = item.get("score", 0.0)
            rerank_score = item.get("rerank_score", 0.0)
            doc_name = main_chunk.doc_name
            log.info(
                f"Source: {doc_name} (Score: {score:.4f}, Rerank Score: {rerank_score:.4f})",
                extra={'log_type': 'DEBUG'}
            )

            initial_context = initial_contexts.get(main_chunk.id)
            if initial_context:
                log.info(f"Initial Context: {initial_context}", extra={'log_type': 'DEBUG'})

            compressed_context = item.get("compressed_context", "")
            log.info(f"Compressed Context: {compressed_context}", extra={'log_type': 'DEBUG'})
    log.info("=" * 56, extra={'log_type': 'INFO'})

def run_cli(reindex: bool, cfg: RAGConfig, user_role: str) -> None:
    """
    The main execution loop for the command-line interface.
    """
    setup_logging()
    pipeline = setup_pipeline(reindex=reindex, cfg=cfg)

    log.info("Enter your question (or 'exit' to quit):", extra={'log_type': 'INFO'})

    while True:
        try:
            query = input("> ").strip()
            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                log.info("Exiting. Goodbye!", extra={'log_type': 'INFO'})
                sys.exit(0)

            log.info(f"Processing query: '{query}'", extra={'log_type': 'USER_QUERY'})
            process_query(query, pipeline, user_role=user_role)

        except (KeyboardInterrupt, EOFError):
            log.info("\nExiting. Goodbye!", extra={'log_type': 'INFO'})
            sys.exit(0)
        except Exception:
            log.exception("An unexpected error occurred.", extra={'log_type': 'ERROR'})


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="A command-line interface for the RAG pipeline.")
    parser.add_argument("--user-role", type=str, help="User name")
    parser.add_argument("--password", type=str, help="User password")
    parser.add_argument(
        '--reindex',
        action='store_true',
        help='If set, the document index will be rebuilt before starting.'
    )
    args = parser.parse_args()

    cfg = RAGConfig()

    username, user_role = authenticate_user(
        user_role_arg=args.user_role,
        password_arg=args.password,
        default_role=cfg.default_role,
        acl_enabled=cfg.acl,
    )

    log.info(
        "Authenticated user='%s' role='%s'",
        username,
        user_role,
        extra={'log_type': 'INFO'}
    )

    run_cli(reindex=args.reindex, cfg=cfg, user_role=user_role)