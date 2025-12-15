from __future__ import annotations

import logging
from elasticsearch import Elasticsearch

ES_URL = "http://127.0.0.1:9200"

def get_es() -> Elasticsearch:
    return Elasticsearch(ES_URL)

def check_es_or_die(es: Elasticsearch) -> None:
    try:
        info = es.info()
    except Exception as e:
        raise RuntimeError(
            "Elasticsearch недоступен. "
            "Проверь, что контейнер запущен и порт 9200 открыт."
        ) from e

    return True
