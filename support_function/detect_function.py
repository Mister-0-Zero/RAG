"""
Provides helper functions for detecting language and category from text.
"""

from pathlib import Path

def detect_category(text: str | Path) -> str | None:
    """
    Detects a category from text or file path by searching for keywords.
    """
    if isinstance(text, Path):
        text = text.name  # или text.as_posix(), если нужно учитывать папки

    text = text.lower()

    if "gate" in text or "ворота" in text:
        return "gate"
    if "channel" in text or "канал" in text:
        return "channel"
    if "center" in text or "центр" in text:
        return "center"
    return "general"

def detect_language(text: str) -> str | None:
    """Detects if text is primarily Russian, English, or mixed."""
    cyrillic_chars = sum(1 for char in text if 'а' <= char <= 'я' or 'А' <= char <= 'Я')
    latin_chars = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z')

    if cyrillic_chars == 0 and latin_chars == 0:
        return "mixed"

    relationship = min(cyrillic_chars, latin_chars) / max(cyrillic_chars, latin_chars)

    if relationship > 0.6:
        return "mixed"
    elif cyrillic_chars > latin_chars:
        return "ru"
    elif latin_chars > cyrillic_chars:
        return "en"
    else:
        return "mixed"