"""
This module provides a centralized, color-coded logging setup for the application.
It uses the colorama library to differentiate log messages by type and level,
improving readability in the console output.
"""
import logging
import sys
from colorama import Fore, Style, init

_LOG_MODE = 0


def set_log_mode(mode: int) -> None:
    global _LOG_MODE
    try:
        _LOG_MODE = int(mode)
    except (TypeError, ValueError):
        _LOG_MODE = 0


class LogModeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        log_type = getattr(record, "log_type", record.levelname)

        if record.levelno >= logging.WARNING:
            return True

        if log_type in {"USER_QUERY", "MODEL_RESPONSE"}:
            return True

        if log_type == "ENHANCEMENT":
            return bool(_LOG_MODE & 1)

        if log_type in {"CONTEXT_AFTER_RERANK", "CONTEXT_BEFORE", "CONTEXT_AFTER"}:
            return bool(_LOG_MODE & 4)

        if log_type in {"INFO", "METADATA", "DEVICE", "DEBUG"}:
            return bool(_LOG_MODE & 2)

        return True

class ColoredFormatter(logging.Formatter):
    """
    A custom log formatter that adds color to log messages based on their level
    and an optional 'log_type' for more granular control.
    """

    LOG_COLORS = {
        'ERROR': Fore.RED,
        'USER_QUERY': Fore.GREEN,
        'INFO': Fore.BLUE,
        'MODEL_RESPONSE': Fore.CYAN,
        'METADATA': Fore.MAGENTA,
        'DEVICE': Fore.YELLOW,
        'ENHANCEMENT': Fore.YELLOW,
        'CONTEXT_AFTER_RERANK': Fore.GREEN,
        'CONTEXT_BEFORE': Fore.YELLOW,
        'CONTEXT_AFTER': Fore.GREEN,
        'DEFAULT': Fore.WHITE,
    }

    def __init__(self, fmt="%(message)s"):
        """Initializes the formatter and colorama."""
        super().__init__(fmt)
        init(autoreset=True)

    def format(self, record):
        """
        Formats the log record with appropriate colors.
        It uses 'log_type' if available, otherwise falls back to the log level name.
        """
        log_type = getattr(record, 'log_type', record.levelname)
        color = self.LOG_COLORS.get(log_type, self.LOG_COLORS['DEFAULT'])

        if log_type in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            log_fmt = f"{Style.BRIGHT}[{record.levelname}]{Style.NORMAL} {self._fmt}"
        else:
            log_fmt = self._fmt

        formatter = logging.Formatter(log_fmt)
        return color + formatter.format(record) + Style.RESET_ALL

def setup_logging(cfg=None, mode: int | None = None):
    """
    Configures the root logger for the application.
    This setup includes:
    - A colored console handler for INFO-level messages.
    - A file handler for DEBUG-level messages.
    - Suppression of excessive logging from third-party libraries.
    """
    # Set higher logging levels for noisy third-party libraries
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    if mode is not None:
        set_log_mode(mode)
    elif cfg is not None:
        set_log_mode(getattr(cfg, "log_mode", 0))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler for readable, colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.addFilter(LogModeFilter())
    root_logger.addHandler(console_handler)

    # File handler for detailed, persistent logs
    file_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(LogModeFilter())
    root_logger.addHandler(file_handler)
