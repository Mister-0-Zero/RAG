"""
This module provides a context compression mechanism using a language model.
"""
from __future__ import annotations

import logging
import requests
from typing import List

from rag.chunking import Chunk
from rag.config import RAGConfig

log = logging.getLogger(__name__)


class ContextCompressor:
    def __init__(self, cfg: RAGConfig, neighbors: int = 5) -> None:
        self.model_name = cfg.model_name_for_compressor
        self.ollama_url = cfg.ollama_url
        self.max_new_tokens = cfg.max_tokens_after_compressed_per_result_ * neighbors
        self.temperature = cfg.temperature_model_compressor

        log.info(
            "Context compressor initialized (ollama model=%s)",
            self.model_name,
        )

    def compress(self, question: str, chunks: List[Chunk]) -> str:
        prompt = self._build_prompt(question, chunks)

        log.debug(
            "Context before compression:\n%s",
            prompt,
            extra={"log_type": "CONTEXT_BEFORE"},
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "repeat_penalty": 1.1,
            },
        }

        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()

        text = resp.json().get("response", "").strip()

        log.debug(
            "Context after compression:\n%s",
            text,
            extra={"log_type": "CONTEXT_AFTER"},
        )

        return text

    def _build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        fragments = []
        for i, ch in enumerate(chunks, 1):
            fragments.append(f"[Фрагмент {i}]\n{ch.text}")

        fragments_text = "\n\n".join(fragments)

        return f"""Ты — система сжатия контекста для Retrieval-Augmented Generation.
Вопрос пользователя:
{question}

Ниже приведен фрагмент текста из базы знаний.
Твоя задача — оставить ТОЛЬКО информацию, необходимую для ответа на вопрос.

Правила:
- Удали явные повторы и технический мусор
- Сохрани связный текст, который помогает понять тему вопроса
- Сохрани определения, пояснения и важные описания, даже если они не отвечают на вопрос напрямую
- Не сокращай фразы до телеграфного стиля — текст должен оставаться читаемым
- НИЧЕГО не добавляй от себя и не делай выводов
- Не отвечай на вопрос, только подготавливай контекст
- Если фрагмент частично полезен — сохрани полезную часть
- Если фрагмент не относится к теме совсем — можешь его опустить


Фрагмент:
{fragments_text}

Формат ответа:
<сжатый контекст>

Сжатый контекст:
"""