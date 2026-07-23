"""Text preprocessing utilities for FAQ matching."""

from __future__ import annotations

import re
from typing import Iterable, List


TYPO_CORRECTIONS = {
    "recieve": "receive",
    "teh": "the",
    "studnet": "student",
    "registation": "registration",
    "enroll": "enroll",
    "tution": "tuition",
    "examn": "exam",
    "hostell": "hostel",
    "libary": "library",
    "grammer": "grammar",
}

PHRASE_ALIASES = {
    "how can i apply": "admission application",
    "how do i enroll": "register course",
    "where do i pay": "fee payment",
    "when are the exams": "exam schedule",
    "how do i get a room": "hostel room",
    "where is the library": "library service",
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        word_tokenize("hello world")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        WordNetLemmatizer().lemmatize("running")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover - fallback for restricted environments
    _STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "to", "of", "in",
        "on", "for", "with", "without", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "those", "these", "i", "you", "we", "they", "he", "she",
        "my", "our", "their", "me", "us", "him", "her", "do", "does", "did", "have", "has",
        "had", "can", "could", "should", "would", "will", "may", "might", "must", "from",
        "by", "at", "as", "into", "about", "after", "before", "during", "while", "how", "what",
        "why", "when", "where", "who", "whom", "which", "how", "please"
    }
    _LEMMATIZER = None


def tokenize_text(text: str) -> List[str]:
    """Tokenize a string into words."""
    if not text:
        return []
    if _LEMMATIZER is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def _correct_typos(text: str) -> str:
    """Apply a small typo correction pass before tokenization."""
    corrected = text.lower()
    for typo, replacement in TYPO_CORRECTIONS.items():
        corrected = re.sub(rf"\b{typo}\b", replacement, corrected)
    return corrected


def preprocess_text(text: str) -> str:
    """Clean and normalize a text passage for NLP matching."""
    if not text:
        return ""

    text = _correct_typos(text)
    for alias, replacement in PHRASE_ALIASES.items():
        if alias in text:
            text = text.replace(alias, replacement)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = tokenize_text(text)
    filtered_tokens = []
    for token in tokens:
        if token in _STOP_WORDS or len(token) <= 1:
            continue
        if _LEMMATIZER is not None:
            token = _LEMMATIZER.lemmatize(token)
            if token in {"requirement", "requirements"}:
                token = "requirements"
        filtered_tokens.append(token)

    return " ".join(filtered_tokens)


def preprocess_batch(texts: Iterable[str]) -> List[str]:
    """Apply preprocessing to a collection of texts."""
    return [preprocess_text(text) for text in texts]
