"""
This module provides a context compression mechanism using a language model.
"""
from __future__ import annotations

import logging
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.chunking import Chunk
from rag.config import RAGConfig

log = logging.getLogger(__name__)


class ContextCompressor:
    """
    Manages the compression of text chunks into a more concise context using a pre-trained language model.
    """
    def __init__(
        self,
        cfg: RAGConfig,
        neighbors: int = 5,
    ) -> None:
        """
        Initializes the ContextCompressor, loading the specified language model and tokenizer.
        """
        log.info("Loading compressor model: %s", cfg.model_name_for_compressor)
        self.device = cfg.device
        self.max_new_tokens = cfg.max_tokens_after_compressed_per_result_ * neighbors
        self.temperature = cfg.temperature_model_compressor
        self.model_name = cfg.model_name_for_compressor

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

    def compress(self, question: str, chunks: List[Chunk]) -> str:
        """
        Compresses a list of text chunks relevant to a given question using the loaded language model.
        """
        prompt = self._build_prompt(question, chunks)
        log.info(f"Context before compression:\n{prompt}", extra={"log_type": "CONTEXT_BEFORE"})

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        
        log.info(f"Context after compression:\n{text.strip()}", extra={"log_type": "CONTEXT_AFTER"})
        return text.strip()

    def _build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        """
        Constructs the prompt string for the language model based on the question and provided chunks.
        """
        fragments = []
        for i, ch in enumerate(chunks, 1):
            fragments.append(f"[Фрагмент {i}]\n{ch.text}")

        fragments_text = "\n\n".join(fragments)

        return f"""Ты — система сжатия контекста для Retrieval-Augmented Generation. Вопрос пользователя:
                    {question}

                    Ниже приведен фрагмент текста из базы знаний.
                    Твоя задача — оставить ТОЛЬКО информацию, необходимую для ответа на вопрос.

                    Правила:
                    - Удали повторы и неинформативные части
                    - Сохрани только важные факты, относящиеся к вопросу
                    - Сохрани определения, ключевые свойства, факты
                    - НИЧЕГО не добавляй от себя
                    - Не отвечай на вопрос, только сжимай текст
                    - Если фрагмент не содержит полезной информации для ответа на вопрос, ничего не добавляй.

                    Формат ответа:
                    <сжатый контекст>

                    Фрагмент:
                    {fragments_text}

                    Сжатый контекст: """