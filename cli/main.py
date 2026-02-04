"""Command-line interface for interacting with the RAG pipeline."""
from __future__ import annotations

import sys
import logging
import argparse
from typing import cast
from dotenv import load_dotenv

from rag.config import RAGConfig
from rag.logger import setup_logging
from rag.api import build_pipeline, process_query, log_final_result, QueryDebug
from rag.answer import AnswerResult
from rag.auth import authenticate_user

log = logging.getLogger(__name__)


def run_cli(reindex: bool, cfg: RAGConfig, user_role: str) -> None:
    """
    The main execution loop for the command-line interface.
    """
    setup_logging(cfg)
    pipeline = build_pipeline(reindex=reindex, cfg=cfg)

    log.info("Enter your question (or 'exit' to quit):", extra={"log_type": "INFO"})

    while True:
        try:
            query = input("> ").strip()
            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                log.info("Exiting. Goodbye!", extra={"log_type": "INFO"})
                sys.exit(0)

            log.info("Processing query: '%s'", query, extra={"log_type": "USER_QUERY"})
            answer_result, debug = cast(
                tuple[AnswerResult, QueryDebug],
                process_query(query, pipeline, user_role=user_role, return_debug=True),
            )
            log_final_result(answer_result, debug.initial_contexts, pipeline.cfg)

        except (KeyboardInterrupt, EOFError):
            log.info("\nExiting. Goodbye!", extra={"log_type": "INFO"})
            sys.exit(0)
        except Exception:
            log.exception("An unexpected error occurred.", extra={"log_type": "ERROR"})


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="A command-line interface for the RAG pipeline.")
    parser.add_argument("--user-role", type=str, help="User name")
    parser.add_argument("--password", type=str, help="User password")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="If set, the document index will be rebuilt before starting.",
    )
    args = parser.parse_args(argv)

    cfg = RAGConfig()

    username, user_role = authenticate_user(
        user_role_arg=args.user_role,
        password_arg=args.password,
        default_role=cfg.default_role,
        acl_enabled=cfg.acl,
    )

    log.info(
        "Authenticated user='%s' role='%s'",
        username,
        user_role,
        extra={"log_type": "INFO"},
    )

    run_cli(reindex=args.reindex, cfg=cfg, user_role=user_role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
