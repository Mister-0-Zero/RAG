from __future__ import annotations

import atexit
import subprocess
import time
from pathlib import Path

from elasticsearch import Elasticsearch


def _run(cmd: str, cwd: Path | None = None) -> None:
    # shell=True хорошо работает на Windows, docker-compose/dockerd в PATH
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=True, check=True)


def _is_es_ready(url: str = "http://localhost:9200", timeout_s: float = 0.3) -> bool:
    try:
        es = Elasticsearch(url, request_timeout=timeout_s)
        return bool(es.ping())
    except Exception:
        return False


def ensure_elasticsearch(
    compose_dir: Path,
    url: str = "http://localhost:9200",
    wait_seconds: int = 30,
) -> bool:
    """
    Поднимает Elasticsearch через docker compose, если он не доступен.
    Возвращает True, если мы его подняли сами (и значит должны потом опустить).
    """
    if _is_es_ready(url):
        return False  # уже запущен

    _run("docker compose up -d", cwd=compose_dir)

    # ждём готовности
    start = time.time()
    while time.time() - start < wait_seconds:
        if _is_es_ready(url):
            return True
        time.sleep(0.5)

    raise RuntimeError("Elasticsearch не поднялся: timeout ожидания ping()")


def shutdown_elasticsearch(compose_dir: Path) -> None:
    # down (вместе с сетью). Если хочешь оставлять volume — он по умолчанию не удаляется.
    _run("docker compose down", cwd=compose_dir)


class ESServiceGuard:
    def __init__(self, compose_dir: Path, url: str = "http://localhost:9200") -> None:
        self.compose_dir = compose_dir
        self.url = url
        self._started_by_me = False

    def __enter__(self) -> "ESServiceGuard":
        self._started_by_me = ensure_elasticsearch(self.compose_dir, self.url)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._started_by_me:
            shutdown_elasticsearch(self.compose_dir)
