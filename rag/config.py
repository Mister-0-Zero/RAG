"""
This module defines the configuration for the RAG pipeline using Pydantic.
It centralizes all key parameters, making them accessible and manageable from one place.
"""
from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import torch
import logging

log = logging.getLogger(__name__)

class RAGConfig(BaseModel):
    """
    A Pydantic model that holds the configuration for the RAG application.
    It automatically determines the compute device (CUDA or CPU) on initialization.
    """
    data_raw: Path = Path("data/raw")
    """The path to the directory containing raw data for ingestion."""

    embeddings_model: str = "intfloat/multilingual-e5-base"
    """The name of the SentenceTransformers model for creating embeddings."""

    chroma_db: Path = Path("data/vector_db")
    """The path to the directory where the Chroma vector database is stored."""

    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """The name of the cross-encoder model used for reranking search results."""

    model_name_for_compressor: str = "qwen2.5:3b-instruct"
    """The name of the large language model used for context compression."""

    max_tokens_after_compressed_per_result_: int = 256
    """The maximum number of tokens to keep for each result after compression."""

    temperature_model_compressor: float = 0.0
    """The temperature setting for the compressor model to control randomness."""

    ollama_url: str = "http://localhost:11435"
    """The URL for the Ollama API endpoint."""

    device: str | None = None
    """The compute device ('cuda' or 'cpu') to be used for model inference."""

    local_or_API_model: str = "API"
    """We will use the local or remote model. ('local' or 'API')"""

    api_model_name: str = "llama-3.3-70b-versatile"
    """The name of the model to use via the API."""
    api_temperature: float = 0.5
    """The temperature for the API model, controlling creativity."""
    api_timeout_s: int = 5
    """The timeout in seconds for API requests."""
    api_max_tokens: int = 2048
    """The maximum number of tokens for the API model to generate."""

    local_model_name: str = "deepseek-r1:32b"
    """The name of the local model to use via Ollama."""
    local_temperature: float = 0.5
    """The temperature for the local model."""
    local_timeout_s: float = 30
    """The timeout in seconds for local model requests."""
    local_max_tokens: int = 2048
    """The maximum number of tokens for the local model to generate."""

    no_data_response: str = "Данных в базе знаний не нашлось."
    """The default response when no relevant information is found."""
    enable_citations: bool = False
    """Whether to include citations in the answer."""

    def model_post_init(self, __context):
        """
        Determines the available compute device after the model is initialized.
        It checks for CUDA availability and sets the device accordingly,
        then logs the result.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Compute device set to: {self.device.upper()}", extra={"log_type": "DEVICE"})
