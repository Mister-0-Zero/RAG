"""
This module provides clients for interacting with Large Language Models (LLMs).
It includes a base client and specific implementations for different LLM providers like Groq and Ollama.
The main purpose is to abstract the details of API calls and provide a unified interface for text generation.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

import requests

from rag.config import RAGConfig

log = logging.getLogger(__name__)


def _get_env_api_key() -> str:
    """
    Retrieves the API key from environment variables.
    It checks for 'MODEL_API_KEY' and raises an error if it's not set.
    """
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise RuntimeError("API key not set. Set MODEL_API_KEY in environment/.env")
    return key


class LLMClient:
    """
    An abstract base class for LLM clients.
    It defines the interface for text generation, ensuring that all subclasses
    provide a consistent method for generating text from a prompt. This class
    is intended for text generation only and does not include any RAG-specific logic.
    """
    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        """
        Generates text based on a given prompt.
        Subclasses must implement this method.
        """
        raise NotImplementedError


class GroqLLMClient(LLMClient):
    """
    A client for the Groq API, which is compatible with OpenAI's Chat Completions format.
    This client handles the specifics of forming requests, including authentication and payload structure,
    and processes the response to extract the generated text.
    """
    def __init__(self, cfg: RAGConfig):
        """Initializes the GroqLLMClient with the given configuration."""
        self.api_key = _get_env_api_key()
        self.base_url = getattr(cfg, "api_base_url", "https://api.groq.com/openai/v1")
        self.model_name = cfg.api_model_name
        self.temperature = getattr(cfg, "api_temperature")
        self.max_tokens = getattr(cfg, "api_max_tokens")
        self.timeout_s = getattr(cfg, "api_timeout_s")

        log.info(
            "Groq LLM init: base_url=%s model=%s temp=%s max_tokens=%s timeout=%ss",
            self.base_url, self.model_name, self.temperature, self.max_tokens, self.timeout_s,
            extra={'log_type': 'INFO'}
        )

    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        """
        Sends a request to the Groq API to generate text and returns the response.
        It constructs the request payload in the OpenAI-compatible format and handles API errors.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("Groq request failed: %s", e, extra={'log_type': 'ERROR'})
            raise

        dt = time.time() - t0
        log.info("Groq response: status=%s time=%.2fs", resp.status_code, dt, extra={'log_type': 'MODEL_RESPONSE'})

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content or not content.strip():
            log.error("Groq returned empty content. Raw: %s", data, extra={'log_type': 'ERROR'})
            raise RuntimeError(f"Groq returned empty content. Raw: {data}")

        return content.strip()


class OllamaLLMClient(LLMClient):
    """
    A client for the Ollama API, which allows running local language models.
    This client is responsible for sending requests to the local Ollama server
    and parsing the response to get the generated text.
    """
    def __init__(self, cfg: RAGConfig):
        """Initializes the OllamaLLMClient with the given configuration."""
        self.base_url = cfg.ollama_url
        self.model_name = cfg.local_model_name
        self.temperature = getattr(cfg, "local_temperature", 0.3)
        self.max_tokens = getattr(cfg, "local_max_tokens", 800)
        self.timeout_s = getattr(cfg, "local_timeout_s", 120)

        log.info(
            "Ollama LLM init: url=%s model=%s", self.base_url, self.model_name,
            extra={'log_type': 'INFO'}
            )

    def generate(self, prompt: str, lang: Optional[str] = None) -> str:
        """
        Sends a request to the local Ollama server to generate text.
        It includes parameters for temperature and token limits, tailored to the Ollama API.
        """
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("Ollama request failed: %s", e, extra={'log_type': 'ERROR'})
            raise

        dt = time.time() - t0
        log.info("Ollama response: status=%s time=%.2fs", resp.status_code, dt, extra={'log_type': 'MODEL_RESPONSE'})

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            log.error("Ollama returned empty content. Raw: %s", data, extra={'log_type': 'ERROR'})
            raise RuntimeError(f"Ollama returned empty content. Raw: {data}")
        return content.strip()


def init_llm_client(cfg: RAGConfig) -> LLMClient:
    """
    Initializes and returns the appropriate LLM client based on the application configuration.
    This factory function reads the 'local_or_API_model' setting from the config
    and instantiates either a 'GroqLLMClient' for API-based models or an 'OllamaLLMClient' for local models.
    """
    mode = cfg.local_or_API_model
    log.info(f"Initializing LLM client in '{mode}' mode.", extra={'log_type': 'INFO'})
    if mode == "API":
        return GroqLLMClient(cfg)
    if mode == "local":
        return OllamaLLMClient(cfg)
    raise ValueError(f"local_or_API_model must be 'API' or 'local', got: {mode}")
