"""Entry point for the FAQ chatbot application."""

from pathlib import Path

from faq_chatbot.utils.chatbot import FAQChatbot


def main() -> None:
    faq_path = Path(__file__).resolve().parent / "faq_chatbot" / "data" / "faq.csv"
    chatbot = FAQChatbot(faq_path=faq_path)
    chatbot.run_console()


if __name__ == "__main__":
    main()
