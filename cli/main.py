"""
This module provides a command-line interface for interacting with the RAG pipeline.
It handles argument parsing, sets up logging, initializes the RAG components,
and processes user queries.
"""
from __future__ import annotations

import sys
import logging

from rag.config import RAGConfig
from rag.compressor import ContextCompressor
from rag.logger import setup_logging
from rag.pipeline import build_hybrid_retriever
from rag.rerank import Reranker
from search.es_client  import get_es, check_es_or_die
from support_function.detect_function import *
import argparse

log = logging.getLogger(__name__)

def parse_args():
    """
    Parses command-line arguments for the RAG CLI.
    """
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument('--reindex', action='store_true', help='Reindex the documents before starting the CLI')
    return parser.parse_args()

def main() -> None:
    """
    Main function for the RAG CLI. It sets up the pipeline, processes user input,
    and returns relevant information based on the RAG model.
    """
    args = parse_args()
    reindex = args.reindex

    setup_logging()

    es = get_es()
    check_es_or_die(es)
    neighbors = 3

    cfg = RAGConfig()
    log.info("We are starting to build a hybrid retriever", extra={'log_type': 'INFO'})
    retriever = build_hybrid_retriever(cfg=cfg, chunk_size=600, overlap=150, reindex=reindex)
    log.info("Готово.", extra={'log_type': 'INFO'})

    log.info("Введите вопрос (или exit):", extra={'log_type': 'INFO'})

    compressor = ContextCompressor(cfg)

    while True:
        query = input("> ").strip()
        if not query:
            continue

        log.info(f"Вопрос: {query}", extra={'log_type': 'USER_QUERY'})

        if query.lower() in {"exit", "quit"}:
            log.info("До встречи.", extra={'log_type': 'INFO'})
            sys.exit(0)

        language = detect_language(query)
        category = detect_category(query)

        log.info(f"Язык: {language}, Категория: {category}", extra={'log_type': 'METADATA'})

        results = retriever.retrieve(query, language=language, category=category)

        reranker = Reranker()
        results = reranker.rerank(query, results, top_k=3)
        final = []

        if not results:
            log.error("Ничего не нашлось.\n")
            continue

        for r in results:
            context = retriever._dense._store.get_neighbors(r["main_chunk"], neighbors_forward=neighbors)

            score = r.get("score", 0.0)
            dense_score = r.get("dense_score", 0.0)
            lexical_score = r.get("lexical_score", 0.0)
            lexical_norm = r.get("lexical_norm", 0.0)
            rerank_score = r.get("rerank_score", 0.0)
            doc_name = r['main_chunk'].doc_name
            main_chunk = r["main_chunk"]
            text = [r.text + " " for r in context]
            log.info(f"=== Результат (score={score:.4f}, rerank={rerank_score:.4f}, dense={dense_score:.4f}, lexical={lexical_score:.4f},\
                      lexical_norm={lexical_norm:.4f}) ===", extra={'log_type': 'MODEL_RESPONSE'})
            log.info(f"Источник: {doc_name}, doc_id: {main_chunk.doc_id}, Категория: {main_chunk.category}, Язык: {main_chunk.language}", extra={'log_type': 'METADATA'})
            log.info(f"Контекст до сжатия: {text}", extra={'log_type': 'MODEL_RESPONSE'})
            log.info("-" * 10, extra={'log_type': 'MODEL_RESPONSE'})

            compressed_context = compressor.compress(question=query, chunks=context)
            final.append({**r, "compressed_context": compressed_context})


        log.info("=== Сжатый контекст ===", extra={'log_type': 'MODEL_RESPONSE'})
        for item in final:
            log.info(item["compressed_context"] + "\n\n", extra={'log_type': 'MODEL_RESPONSE'})



if __name__ == "__main__":
    main()