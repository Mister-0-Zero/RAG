def detect_category(text: str) -> str | None:
    text = text.lower()
    if "gate" in text or "ворота" in text:
        return "gate"
    if "channel" in text or "канал" in text:
        return "channel"
    if "center" in text or "центр" in text:
        return "center"
    return "general"

def detect_language(text: str) -> str | None:
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