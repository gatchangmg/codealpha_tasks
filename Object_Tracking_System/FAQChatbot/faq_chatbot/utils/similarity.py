"""TF-IDF based FAQ retrieval utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SYNONYM_MAP = {
    "admission": ["admission", "admissions", "apply", "application", "enroll", "enrollment"],
    "register": ["register", "registration", "enroll", "course", "courses"],
    "fee": ["fee", "fees", "payment", "tuition", "charge", "charges"],
    "exam": ["exam", "examination", "exams", "test", "assessment"],
    "graduate": ["graduate", "graduation", "degree", "certificate"],
    "library": ["library", "book", "books", "resource", "resources"],
    "hostel": ["hostel", "dorm", "residence", "room"],
    "campus": ["campus", "facility", "facilities", "building", "buildings"],
}

from faq_chatbot.utils.preprocessing import preprocess_text


def build_tfidf_model(questions: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Train a TF-IDF model on FAQ questions."""
    processed_questions = [preprocess_text(question) for question in questions]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, matrix


def _keyword_overlap_score(query_tokens: List[str], faq_tokens: List[str]) -> float:
    """Compute a lightweight overlap score using domain synonyms."""
    if not query_tokens or not faq_tokens:
        return 0.0

    query_set = set(query_tokens)
    faq_set = set(faq_tokens)
    direct_matches = len(query_set & faq_set)
    expanded_matches = 0

    for token in query_set:
        for canonical, variants in SYNONYM_MAP.items():
            if token == canonical or token in variants:
                if canonical in faq_set or any(variant in faq_set for variant in variants):
                    expanded_matches += 1
                    break

    total = direct_matches + expanded_matches
    return total / max(len(query_set), 1)


def find_best_match(
    user_question: str,
    questions: List[str],
    answers: List[str],
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    threshold: float = 0.35,
) -> Tuple[Optional[dict], float]:
    """Return the best FAQ match and confidence score for a user question."""
    if not questions:
        return None, 0.0

    processed_query = preprocess_text(user_question)
    query_tokens = processed_query.split()
    query_vector = vectorizer.transform([processed_query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    faq_scores = []
    for faq_question in questions:
        faq_tokens = preprocess_text(faq_question).split()
        overlap = _keyword_overlap_score(query_tokens, faq_tokens)
        faq_scores.append(overlap)

    overlap_scores = np.array(faq_scores)
    combined_scores = 0.65 * scores + 0.35 * overlap_scores
    best_index = int(np.argmax(combined_scores))
    best_score = float(combined_scores[best_index])

    if best_score < threshold:
        return None, best_score

    return {
        "question": questions[best_index],
        "answer": answers[best_index],
        "score": best_score,
    }, best_score
