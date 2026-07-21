"""Console and Streamlit chatbot interface."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from faq_chatbot.utils.preprocessing import preprocess_text
from faq_chatbot.utils.similarity import build_tfidf_model, find_best_match


class FAQChatbot:
    """An NLP FAQ chatbot that answers university-related questions."""

    def __init__(self, faq_path: str | Path, threshold: float = 0.35) -> None:
        self.faq_path = Path(faq_path)
        self.threshold = threshold
        self.logger = self._configure_logger()
        self.faq_df = self._load_faq_data()
        self.vectorizer, self.matrix = build_tfidf_model(self.faq_df["Question"].tolist())
        self.chat_history: List[dict] = []

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("faq_chatbot")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _load_faq_data(self) -> pd.DataFrame:
        try:
            faq_df = pd.read_csv(self.faq_path)
            required_columns = {"Question", "Answer"}
            missing = required_columns.difference(faq_df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return faq_df[["Question", "Answer"]].copy()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"FAQ dataset not found at {self.faq_path}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load FAQ dataset: {exc}") from exc

    def _log_interaction(self, user_input: str, answer: str, score: float) -> None:
        self.logger.info("User: %s | Bot: %s | Score: %.3f", user_input, answer, score)

    def _append_history(self, user_input: str, answer: str, score: float) -> None:
        self.chat_history.append(
            {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_input,
                "bot": answer,
                "score": round(score, 3),
            }
        )

    def _get_fallback_response(self, user_question: str) -> str:
        """Return a helpful fallback response for greetings or out-of-scope questions."""
        normalized = user_question.strip().lower()
        if normalized in {"hi", "hello", "hey", "hey there", "good morning", "good evening", "good afternoon"}:
            return (
                "Hello! I can help with university admissions, course registration, tuition fees, "
                "examinations, graduation, library services, hostel services, and campus facilities."
            )

        if normalized in {"thanks", "thank you", "bye", "goodbye", "see you"}:
            return "You’re welcome! Ask me about admissions, registration, exams, fees, or campus services."

        if len(normalized.split()) <= 2:
            return (
                "I can help with university-related FAQ topics such as admissions, registration, fees, "
                "exams, graduation, library services, hostel services, and campus facilities."
            )

        return (
            "I’m focused on university FAQ topics. Please ask about admissions, registration, fees, "
            "exams, graduation, library services, hostel services, or campus facilities."
        )

    def get_answer(self, user_question: str) -> Tuple[Optional[str], Optional[dict], float]:
        """Get the best FAQ answer for the user's question."""
        if not user_question.strip():
            return "Please enter a question.", None, 0.0

        processed_query = preprocess_text(user_question)
        if not processed_query:
            return self._get_fallback_response(user_question), None, 0.0

        try:
            match, score = find_best_match(
                user_question,
                self.faq_df["Question"].tolist(),
                self.faq_df["Answer"].tolist(),
                self.vectorizer,
                self.matrix,
                threshold=self.threshold,
            )
        except Exception as exc:
            self.logger.exception("Similarity matching failed: %s", exc)
            return "Sorry, I couldn't process that question right now.", None, 0.0

        if match is None:
            answer = self._get_fallback_response(user_question)
            self._log_interaction(user_question, answer, score)
            self._append_history(user_question, answer, score)
            return answer, None, score

        response = match["answer"]
        self._log_interaction(user_question, response, match["score"])
        self._append_history(user_question, response, match["score"])
        return response, match, match["score"]

    def export_chat_history(self, path: str | Path) -> None:
        """Export the conversation history to JSON."""
        output_path = Path(path)
        output_path.write_text(json.dumps(self.chat_history, indent=2), encoding="utf-8")

    def run_console(self) -> None:
        """Launch a simple console chat interface."""
        print("\n---------------------------------")
        print("FAQ Chatbot")
        print("---------------------------------")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Bot: Goodbye!")
                break
            answer, _, _ = self.get_answer(user_input)
            print(f"Bot: {answer}\n")
