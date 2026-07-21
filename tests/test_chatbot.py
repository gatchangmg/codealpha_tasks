import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faq_chatbot.utils.chatbot import FAQChatbot
from faq_chatbot.utils.preprocessing import preprocess_text
from faq_chatbot.utils.similarity import build_tfidf_model, find_best_match


def test_preprocess_text_removes_stopwords_and_normalizes():
    text = "What are the admission requirements?"
    cleaned = preprocess_text(text)

    assert "admission" in cleaned
    assert "requirements" in cleaned
    assert "what" not in cleaned
    assert "are" not in cleaned


def test_find_best_match_returns_expected_faq():
    questions = [
        "What are the admission requirements?",
        "How do I register for courses?",
    ]
    answers = ["You need academic certificates and an application form.", "Use the student portal to register."]

    vectorizer, matrix = build_tfidf_model(questions)
    result, score = find_best_match(
        "What are the admission requirements?",
        questions,
        answers,
        vectorizer,
        matrix,
        threshold=0.0,
    )

    assert result is not None
    assert result["question"] == questions[0]
    assert score >= 0.0


def test_greeting_returns_helpful_message():
    chatbot = FAQChatbot(Path("faq_chatbot/data/faq.csv"))
    answer, match, _ = chatbot.get_answer("hi")

    assert match is None
    assert "university" in answer.lower()


def test_out_of_domain_question_returns_scope_message():
    chatbot = FAQChatbot(Path("faq_chatbot/data/faq.csv"))
    answer, match, _ = chatbot.get_answer("what is ai")

    assert match is None
    assert "admissions" in answer.lower() or "registration" in answer.lower()
