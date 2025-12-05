from __future__ import annotations

import sys

from rag.config import RAGConfig
from rag.pipeline import build_dense_retriever


def main() -> None:
    cfg = RAGConfig()
    print("Строим dense-ретривер (ingest → chunk → index)...")
    retriever, chunks = build_dense_retriever(cfg=cfg)

    print(f"Готово. Документов: {len(set(c.doc_id for c in chunks))}, чанков: {len(chunks)}")
    print("Введите вопрос (или exit):")

    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("До встречи.")
            sys.exit(0)

        results = retriever.retrieve(query, top_k=3)

        if not results:
            print("Ничего не нашлось.\n")
            continue

        for i, r in enumerate(results, start=1):
            chunk = r["chunk"]
            distance = r["distance"]
            print(f"\n=== Результат {i} (distance={distance:.4f}) ===")
            print(chunk.text[:500])
            print("--------")

        print()

if __name__ == "__main__":
    main()