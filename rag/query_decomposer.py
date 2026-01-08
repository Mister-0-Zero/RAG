import logging
import re
from typing import List

log = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


class QueryDecomposer:
    """
    Decomposition of a complex user query into subqueries.
    Used ONLY before retrieval.
    """

    def __init__(self, llm_client, cfg):
        self.llm = llm_client
        self.cfg = cfg
        self.min_words = cfg.min_words_for_decomposition

    def decompose(self, query: str) -> List[str]:
        wc = _word_count(query)

        if wc <= self.min_words:
            log.debug("Query short (%d words). No decomposition.", wc)
            return [query]

        prompt = self._build_prompt(query)

        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            log.exception("Query decomposition failed, fallback to original query: %s", e)
            return [query]

        subqueries = self._parse_response(raw)

        if not subqueries:
            log.warning("Empty decomposition result, fallback to original query.")
            return [query]

        log.info("Decomposed query into %d subqueries", len(subqueries))
        return subqueries

    def _build_prompt(self, query: str) -> str:
        return (
            "Разбей пользовательский запрос на отдельные смысловые подзапросы.\n"
            "Каждый подзапрос должен быть самостоятельным и пригодным для поиска.\n"
            "Если запрос уже простой — верни его без изменений.\n\n"
            "Формат ответа:\n"
            "- один подзапрос на строку\n"
            "- без нумерации\n"
            "- без пояснений\n\n"
            f"Запрос:\n{query}\n\n"
            "Подзапросы:"
        )

    def _parse_response(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.splitlines()]
        lines = [l for l in lines if l]

        clean = []
        for l in lines:
            if len(l) < 3:
                continue
            clean.append(l)

        seen = set()
        result = []
        for q in clean:
            if q not in seen:
                seen.add(q)
                result.append(q)

        return result
