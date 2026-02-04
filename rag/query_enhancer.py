"""
Query enhancement with variations and a hypothetical answer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from support_function.detect_function import detect_language

log = logging.getLogger(__name__)


class QueryEnhancer:
    """
    Generates query variations and an optional hypothetical answer for retrieval.
    """

    def __init__(self, llm_client, cfg):
        self.llm = llm_client
        self.cfg = cfg
        self.variations_count = max(0, int(cfg.query_variations_count))
        self.use_hypothetical = bool(cfg.query_use_hypothetical_answer)

    def enhance(self, query: str) -> tuple[list[str], Optional[str]]:
        if self.variations_count <= 0 and not self.use_hypothetical:
            return [], None

        target_lang = detect_language(query)
        prompt = self._build_prompt(query)

        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            log.warning("Query enhancement failed, fallback to original query: %s", e)
            return [], None

        payload = self._parse_json(raw)
        if not payload:
            log.warning("Query enhancement returned invalid JSON, skipping enhancements.")
            return [], None

        variations = payload.get("variations", [])
        if not isinstance(variations, list):
            variations = []

        clean_variations: list[str] = []
        seen = set()
        for v in variations:
            if not isinstance(v, str):
                continue
            text = v.strip()
            if len(text) < 3:
                continue
            if not self._matches_language(target_lang, text):
                continue
            if text not in seen:
                seen.add(text)
                clean_variations.append(text)

        if self.variations_count:
            clean_variations = clean_variations[: self.variations_count]

        hypo = payload.get("hypothetical_answer") if self.use_hypothetical else None
        if not isinstance(hypo, str):
            hypo = None
        if hypo is not None:
            hypo = hypo.strip() or None
        if hypo and not self._matches_language(target_lang, hypo):
            hypo = None

        return clean_variations, hypo

    def _build_prompt(self, query: str) -> str:
        lang = detect_language(query)
        count = self.variations_count

        if lang == "ru":
            return (
                "Сформируй варианты перефразированного пользовательского запроса для поиска по базе знаний.\n"
                "Смысл должен сохраниться, формулировки должны быть разными.\n"
                "Также сформируй краткий гипотетический ответ (2-4 предложения), если это разрешено.\n\n"
                "Пиши ВСЕ на русском языке.\n"
                "Не используй английский.\n\n"
                "Верни ТОЛЬКО JSON в формате:\n"
                "{\"variations\": [" + ("\"...\"" if count > 0 else "") + "], "
                "\"hypothetical_answer\": \"...\"}\n\n"
                f"Количество вариаций: {count}.\n"
                "Если вариации не нужны, верни пустой список.\n"
                "Если гипотетический ответ не нужен — верни пустую строку.\n\n"
                f"Запрос:\n{query}"
            )

        return (
            "Generate paraphrased variations of the user query for knowledge base retrieval.\n"
            "Keep the meaning, use different wording.\n"
            "Also generate a short hypothetical answer (2-4 sentences) if allowed.\n\n"
            "Write EVERYTHING in the same language as the original query.\n\n"
            "Return ONLY JSON in the format:\n"
            "{\"variations\": [" + ("\"...\"" if count > 0 else "") + "], "
            "\"hypothetical_answer\": \"...\"}\n\n"
            f"Variations count: {count}.\n"
            "If no variations are needed, return an empty list.\n"
            "If a hypothetical answer is not needed, return an empty string.\n\n"
            f"Query:\n{query}"
        )

    def _matches_language(self, target_lang: str, text: str) -> bool:
        if target_lang == "ru":
            return self._has_cyrillic(text)
        if target_lang == "en":
            return not self._has_cyrillic(text)
        return True

    @staticmethod
    def _has_cyrillic(text: str) -> bool:
        return bool(re.search(r"[А-Яа-яЁё]", text))

    def _parse_json(self, text: str) -> dict | None:
        if not text:
            return None

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        raw = text[start : end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
