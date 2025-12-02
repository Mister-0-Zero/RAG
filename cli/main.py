from __future__ import annotations

import sys
from rag.config import RAGConfig


def main() -> None:
    cfg = RAGConfig()

    print("RAG CLI запущен.")
    print("Введите вопрос (или exit):")

    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("До встречи.")
            sys.exit(0)

        # Заглушка — дальше подставим настоящий RAG
        print(f"[DEBUG] Получен запрос: {query}")
        print("Ответ будет тут после интеграции retrieval + LLM.\n")


if __name__ == "__main__":
    main()
