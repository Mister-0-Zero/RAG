from __future__ import annotations

import sys
import logging

from rag.config import RAGConfig
from rag.logger import setup_logging
from rag.pipeline import build_hybrid_retriever
from support_function.detect_function import *

def main() -> None:
    setup_logging()
    cfg = RAGConfig()
    logging.info("Строим dense-ретривер (ingest → chunk → index)...", extra={'log_type': 'INFO'})
    retriever, chunks = build_hybrid_retriever(cfg=cfg, chunk_size=600, overlap=150)

    logging.info(f"Готово. Документов: {len(set(c.doc_id for c in chunks))}, чанков: {len(chunks)}", extra={'log_type': 'INFO'})
    logging.info("Введите вопрос (или exit):", extra={'log_type': 'INFO'})

    while True:
        query = input("> ").strip()
        if not query:
            continue

        logging.info(f"Вопрос: {query}", extra={'log_type': 'USER_QUERY'})

        if query.lower() in {"exit", "quit"}:
            logging.info("До встречи.", extra={'log_type': 'INFO'})
            sys.exit(0)

        language = detect_language(query)
        category = detect_category(query)

        logging.info(f"Язык: {language}, Категория: {category}", extra={'log_type': 'METADATA'})

        results = retriever.retrieve(query, top_k=3, language=language, category=category, neighbors=3)

        if not results:
            logging.error("Ничего не нашлось.\n")
            continue

        for i, r in enumerate(results, start=1):
            score = r.get("score", 0.0)
            dense_score = r.get("score", 0.0)
            lexical_score = r.get("lexical_score", 0.0)
            lexical_norm = r.get("lexical_norm", 0.0)

            doc_name = r['main_chunk'].doc_name
            context_chunks = r['chunk']
            main_chunk = r["main_chunk"]
            full_text = "\n\n".join([c.text for c in context_chunks])

            logging.info(f"=== Результат {i} (score={score:.4f}, dense={dense_score:.4f}, lexical={lexical_score:.4f}, lexical_norm={lexical_norm:.4f}) ===", extra={'log_type': 'MODEL_RESPONSE'})
            logging.info(f"Источник: {doc_name}, doc_id: {main_chunk.doc_id}, Категория: {main_chunk.category}, Язык: {main_chunk.language}", extra={'log_type': 'METADATA'})
            logging.info(f"Фрагмент: {full_text}", extra={'log_type': 'MODEL_RESPONSE'})
            logging.info("-" * 10, extra={'log_type': 'MODEL_RESPONSE'})

        print()

if __name__ == "__main__":
    main()