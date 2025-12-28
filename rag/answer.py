import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    citations: Optional[list[str]] = None


def detect_lang(text: str) -> str:
    cyrillic_chars = sum(1 for char in text if 'а' <= char <= 'я' or 'А' <= char <= 'Я')
    latin_chars = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z')

    if cyrillic_chars >= latin_chars:
        return "ru"
    return "en"


class AnswerGenerator:
    """
    Вход: final (list[dict]) как у тебя:
      item['compressed_context']: str
      item['main_chunk']: Chunk (имеет doc_name)
    """

    def __init__(self, llm_client, cfg):
        self.llm = llm_client
        self.cfg = cfg

    def generate(self, query: str, final: list[dict]) -> AnswerResult:
        lang = detect_lang(query)

        if not final:
            log.warning("No retrieved items. Returning no_data_response.")
            return AnswerResult(answer=self.cfg.no_data_response, citations=None)

        context_text = self._build_context_text(final)
        if not context_text.strip():
            log.warning("Context empty after compression. Returning no_data_response.")
            return AnswerResult(answer=self.cfg.no_data_response, citations=None)

        prompt = self._build_prompt(query=query, context_text=context_text, lang=lang)
        log.info("Promt: \n%s", prompt)

        log.info("Calling LLM for answer. ctx_chars=%d items=%d", len(context_text), len(final))
        answer_text = self.llm.generate(prompt=prompt, lang=lang)

        citations = None
        if self.cfg.enable_citations:
            citations = self._extract_doc_names(final)

        return AnswerResult(answer=answer_text, citations=citations)

    def _build_context_text(self, final: list[dict]) -> str:
        parts = []
        for item in final:
            cc = item.get("compressed_context", "")
            if cc and cc.strip():
                parts.append(cc.strip())
        return "\n\n".join(parts)

    def _extract_doc_names(self, final: list[dict]) -> list[str]:
        docs = []
        for item in final:
            chunk = item.get("main_chunk")
            doc_name = getattr(chunk, "doc_name", None)
            if doc_name and doc_name not in docs:
                docs.append(doc_name)
        return docs

    def _build_prompt(self, query: str, context_text: str, lang: str) -> str:
        if lang == "ru":
            rules = (
                "Правила:\n"
                f"1) Отвечай строго по КОНТЕКСТУ.\n"
                f"2) Если ответа нет в КОНТЕКСТЕ — ответь ровно этой фразой: {self.cfg.no_data_response}\n"
                "3) Не выдумывай факты.\n"
            )
        else:
            rules = (
                "Rules:\n"
                "1) Answer strictly using the CONTEXT.\n"
                f"2) If the answer is not in the CONTEXT, reply exactly with: {self.cfg.no_data_response}\n"
                "3) Do not invent facts.\n"
            )

        return (
            f"{rules}\n"
            "CONTEXT:\n"
            f"{context_text}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "ANSWER:"
        )
