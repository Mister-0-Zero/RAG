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

    model_name_for_compressor: str = "qwen2.5:7b-instruct"
    """The name of the large language model used for context compression."""

    max_tokens_after_compressed_per_result_: int = 800
    """The maximum number of tokens to keep for each result after compression."""

    compressor_prompt: str = (
        "Ты — система очистки контекста для Retrieval-Augmented Generation.\n"
        "Вопрос пользователя:\n"
        "{question}\n\n"
        "Ниже приведены фрагменты текста из базы знаний.\n"
        "Твоя задача — вернуть ОДИН связный, осмысленный кусок текста, который относится\n"
        "исключительно к вопросу. Удали повторы и нерелевантное, но сохрани связность.\n"
        "Пиши естественно, одним блоком текста, без списков.\n\n"
        "Правила:\n"
        "- Возвращай один цельный абзац\n"
        "- Удали явные повторы и технический мусор\n"
        "- Удали любые части, которые не относятся к вопросу, даже если они выглядят связными\n"
        "- Сохрани ключевые факты и определения, относящиеся к вопросу\n"
        "- Не добавляй новых сведений и не делай выводов\n\n"
        "Фрагменты:\n"
        "{fragments_text}\n\n"
        "Формат ответа:\n"
        "<один связный абзац>\n\n"
        "Ответ:\n"
    )
    """Prompt for context cleaning/compression."""

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
    api_timeout_s: int = 10
    """The timeout in seconds for API requests."""
    api_max_tokens: int = 2048
    """The maximum number of tokens for the API model to generate."""

    local_model_name: str = "qwen2.5:7b-instruct"
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

    min_words_for_decomposition: int = 5
    """The minimum number of words in a query to trigger decomposition."""

    neighbors_forward: int = 0
    """Number of neighbor chunks to include after the main chunk."""
    neighbors_backward: int = 0
    """Number of neighbor chunks to include before the main chunk."""

    chunking_mode: str = "delimiter"
    """Chunking mode: 'delimiter' or 'size'."""
    chunk_delimiters: list[str] = ["SECTION:"]
    """List of delimiters for delimiter-based chunking."""
    chunk_delimiters_are_regex: bool = False
    """Whether chunk delimiters are regex patterns."""
    chunk_delimiter_included: bool = False
    """Whether to include delimiter text in chunks."""

    log_mode: int = 7
    """Logging mode bitmask. Always logs user prompt and model answer."""

    query_variations_count: int = 3
    """Number of query variations to generate for retrieval."""
    query_use_hypothetical_answer: bool = True
    """Whether to generate a hypothetical answer and use it as a retrieval query."""
    query_enhancement_mode: str = "single"
    """Query enhancement mode: 'single' or 'multi'."""
    query_enhancer_use_local: bool = True
    """Whether to use a local model for query enhancement."""
    query_enhancer_model_name: str | None = None
    """Optional model name for query enhancement (defaults to local model)."""
    query_enhancer_temperature: float = 0.5
    """Temperature for query enhancement generation."""
    query_enhancer_max_tokens: int = 256
    """Max tokens for query enhancement generation."""
    query_enhancer_timeout_s: int = 30
    """Timeout in seconds for query enhancement requests."""

    acl: bool = False
    """Whether to use Access Control Lists (ACL) for allowing to documents."""
    default_allow: bool = True
    """The default permission for documents without explicit ACLs."""
    acl_rules_path: Path = Path("rag/acl_rules.yaml")
    """The path to the ACL rules YAML file."""
    default_role: str = "guest"
    """The default role assigned to users without specific roles."""


    def model_post_init(self, __context):
        """
        Determines the available compute device after the model is initialized.
        It checks for CUDA availability and sets the device accordingly,
        then logs the result.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Compute device set to: {self.device.upper()}", extra={"log_type": "DEVICE"})
