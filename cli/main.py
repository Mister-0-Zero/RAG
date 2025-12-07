from __future__ import annotations

import sys

from rag.config import RAGConfig
from rag.pipeline import build_dense_retriever
from support_function.detect_function import *

def main() -> None:
    cfg = RAGConfig()
    print("Строим dense-ретривер (ingest → chunk → index)...")
    retriever, chunks = build_dense_retriever(cfg=cfg, chunk_size=600, overlap=150)

    print(f"Готово. Документов: {len(set(c.doc_id for c in chunks))}, чанков: {len(chunks)}")
    print("Введите вопрос (или exit):")

    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("До встречи.")
            sys.exit(0)

        language = detect_language(query)
        category = detect_category(query)
        results = retriever.retrieve(query, top_k=3, language=language, category=category, neighbors=5)

        if not results:
            print("Ничего не нашлось.\n")
            continue

        for i, r in enumerate(results, start=1):
            distance = r["distance"]
            print(F"Язык запроса: {language}, категория запроса: {category}")
            print(f"\n=== Результат {i} (distance={distance:.4f}) ===")
            print(f"Документ: {r['chunk']}")
            full_text = "\n\n".join(c.text for c in chunks)
            print(full_text[:1000])
            print("--------")

        print()

if __name__ == "__main__":
    main()