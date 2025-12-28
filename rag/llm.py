import os
import time
import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)


def _get_env_api_key() -> str:
    # Поддержим оба варианта, чтобы у тебя не ломалось от имени переменной
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise RuntimeError("API key not set. Set MODEL_API_KEY in environment/.env")
    return key


class LLMClient:
    """Только генерация текста по готовому prompt (без RAG-логики)."""
    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        raise NotImplementedError


class GroqLLMClient(LLMClient):
    """
    Groq OpenAI-compatible Chat Completions API.
    Base URL: https://api.groq.com/openai/v1
    Endpoint: /chat/completions
    """
    def __init__(self, cfg):
        self.api_key = _get_env_api_key()
        self.base_url = getattr(cfg, "api_base_url", "https://api.groq.com/openai/v1")
        self.model_name = cfg.api_model_name
        self.temperature = getattr(cfg, "api_temperature")
        self.max_tokens = getattr(cfg, "api_max_tokens")
        self.timeout_s = getattr(cfg, "api_timeout_s")

        log.info(
            "Groq LLM init: base_url=%s model=%s temp=%s max_tokens=%s timeout=%ss",
            self.base_url, self.model_name, self.temperature, self.max_tokens, self.timeout_s
        )

    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Chat Completions формат (OpenAI-compatible)
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            log.exception("Groq request failed: %s", e)
            raise

        dt = time.time() - t0
        log.info("Groq response: status=%s time=%.2fs", resp.status_code, dt)

        if resp.status_code != 200:
            # Печатаем полезный кусок для дебага, но не всю простыню
            try:
                err = resp.json()
            except Exception:
                err = {"text": resp.text[:1000]}
            raise RuntimeError(f"Groq API error {resp.status_code}: {err}")

        data = resp.json()
        # Обычно: choices[0].message.content
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content or not content.strip():
            raise RuntimeError(f"Groq returned empty content. Raw: {data}")

        return content.strip()


class OllamaLLMClient(LLMClient):
    """
    Ollama API: POST {ollama_url}/api/chat
    """
    def __init__(self, cfg):
        self.base_url = cfg.ollama_url
        self.model_name = cfg.local_model_name
        self.temperature = getattr(cfg, "local_temperature", 0.3)
        self.max_tokens = getattr(cfg, "local_max_tokens", 800)
        self.timeout_s = getattr(cfg, "local_timeout_s", 120)

        log.info("Ollama LLM init: url=%s model=%s", self.base_url, self.model_name)

    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # num_predict — аналог max_tokens у Ollama
                "num_predict": self.max_tokens,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            log.exception("Ollama request failed: %s", e)
            raise

        dt = time.time() - t0
        log.info("Ollama response: status=%s time=%.2fs", resp.status_code, dt)

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text[:1000]}")

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise RuntimeError(f"Ollama returned empty content. Raw: {data}")
        return content.strip()


def init_llm_client(cfg) -> LLMClient:
    """
    cfg.local_or_API_model: 'API' | 'local'
    """
    mode = cfg.local_or_API_model
    if mode == "API":
        return GroqLLMClient(cfg)
    if mode == "local":
        return OllamaLLMClient(cfg)
    raise ValueError(f"local_or_API_model must be 'API' or 'local', got: {mode}")
