import re

def normalize_text(text):
    """
    Normalizes text by lowercasing and removing extra whitespace.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()