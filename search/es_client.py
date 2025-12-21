"""
Provides client and health check functions for Elasticsearch.
"""
from __future__ import annotations

import logging
from elasticsearch import Elasticsearch

log = logging.getLogger(__name__)

ES_URL = "http://127.0.0.1:9200"

def get_es() -> Elasticsearch:
    """Returns an Elasticsearch client instance."""
    log.info(f"Creating Elasticsearch client for URL: {ES_URL}", extra={'log_type': 'INFO'})
    return Elasticsearch(ES_URL)

def check_es_or_die(es: Elasticsearch) -> None:
    """Checks if the Elasticsearch service is available, otherwise raises a RuntimeError."""
    try:
        info = es.info()
        log.info(f"Successfully connected to Elasticsearch version {info['version']['number']}", extra={'log_type': 'INFO'})
    except Exception as e:
        log.error("Failed to connect to Elasticsearch.", extra={'log_type': 'ERROR'})
        raise RuntimeError(
            "Elasticsearch недоступен. "
            "Проверь, что контейнер запущен и порт 9200 открыт."
        ) from e
