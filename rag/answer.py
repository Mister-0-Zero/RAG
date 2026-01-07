"""
This module defines the AnswerGenerator class, which is responsible for generating
a final answer based on a user's query and the retrieved context.
It uses a language model to synthesize the answer and can include citations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from support_function.detect_function import detect_language

if TYPE_CHECKING:
    from rag.config import RAGConfig
    from rag.llm import LLMClient

log = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """
    A dataclass to hold the result of the answer generation process.
    It contains the generated answer and an optional list of source citations.
    """
    answer: str
    citations: Optional[list[str]] = None
    prompt: Optional[str] = None
    final_context: Optional[list[dict]] = None


class AnswerGenerator:
    """
    This class orchestrates the final step of the RAG pipeline: generating an answer.
    It takes the user's query and the processed retrieval results (final),
    builds a prompt, and uses an LLM client to generate a text-based answer.
    """

    def __init__(self, llm_client: LLMClient, cfg: RAGConfig):
        """
        Initializes the AnswerGenerator.
        Args:
            llm_client: An instance of a class that implements the LLMClient interface.
            cfg: The application's configuration object.
        """
        self.llm = llm_client
        self.cfg = cfg

    def generate(self, query: str, final: list[dict]) -> AnswerResult:
        """
        Generates an answer by synthesizing information from the retrieved context.
        Args:
            query: The user's original query.
            final: A list of dictionaries, where each contains the compressed context
                   and metadata about the source chunk.
        Returns:
            An AnswerResult object containing the answer and other data for logging.
        """
        lang = detect_language(query)

        if not final:
            log.warning("No retrieved items. Returning no_data_response.", extra={'log_type': 'WARNING'})
            return AnswerResult(
                answer=self.cfg.no_data_response,
                citations=None,
                prompt=None,
                final_context=final,
            )

        context_text = self._build_context_text(final)
        if not context_text.strip():
            log.warning("Context is empty after processing. Returning no_data_response.", extra={'log_type': 'WARNING'})
            return AnswerResult(
                answer=self.cfg.no_data_response,
                citations=None,
                prompt=None,
                final_context=final,
            )

        prompt = self._build_prompt(query=query, context_text=context_text, lang=lang)
        if self.cfg.extended_logs:
            log.info("Prompt: \n%s", prompt, extra={'log_type': 'INFO'})

        log.info(
            "Calling LLM for answer. Context chars: %d, Items: %d",
            len(context_text), len(final),
            extra={'log_type': 'INFO'}
        )
        answer_text = self.llm.generate(prompt=prompt, lang=lang)

        citations = self._extract_doc_names(final) if self.cfg.enable_citations else None

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            prompt=prompt,
            final_context=final,
        )

    def _build_context_text(self, final: list[dict]) -> str:
        """Constructs a single string of context from the compressed results."""
        parts = [
            item.get("compressed_context", "").strip()
            for item in final
            if item.get("compressed_context", "").strip()
        ]
        return "\n\n".join(parts)

    def _extract_doc_names(self, final: list[dict]) -> list[str]:
        """Extracts unique document names (citations) from the retrieval results."""
        doc_names = []
        for item in final:
            chunk = item.get("main_chunk")
            doc_name = getattr(chunk, "doc_name", None)
            if doc_name and doc_name not in doc_names:
                doc_names.append(doc_name)
        return doc_names

    def _build_prompt(self, query: str, context_text: str, lang: str) -> str:
        """Builds the final prompt to be sent to the LLM, including rules and context."""
        if lang == "ru":
            rules = (
                "Правила:\n"
                "1) Отвечай строго по предоставленному КОНТЕКСТУ.\n"
                f"2) Если ответ не найден в КОНТЕКСТЕ, используй фразу: {self.cfg.no_data_response}\n"
                "3) Не придумывай и не добавляй информацию, которой нет в тексте.\n"
            )
            question_title = "ВОПРОС"
            answer_title = "ОТВЕТ"
        else:
            rules = (
                "Rules:\n"
                "1) Answer strictly based on the provided CONTEXT.\n"
                f"2) If the answer is not in the CONTEXT, use the phrase: {self.cfg.no_data_response}\n"
                "3) Do not invent or add information that is not in the text.\n"
            )
            question_title = "QUESTION"
            answer_title = "ANSWER"

        return (
            f"{rules}\n"
            "КОНТЕКСТ:\n"
            f"{context_text}\n\n"
            f"{question_title}:\n"
            f"{query}\n\n"
            f"{answer_title}:"
        )
