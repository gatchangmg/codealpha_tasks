"""Streamlit web interface for the FAQ chatbot."""

from pathlib import Path

import streamlit as st

from faq_chatbot.utils.chatbot import FAQChatbot


st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: white !important;
        font-weight: 700;
    }
    .stCaption, .st-emotion-cache-1rqf6m4 {
        color: #e2e8f0 !important;
    }
    .stChatMessageContent {
        border-radius: 12px;
        background: rgba(255,255,255,0.96);
        color: #0f172a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 0.6rem 0.8rem;
    }
    .stTextInput > div > div > input {
        border-radius: 999px;
        padding: 0.7rem 1rem;
    }
    .stButton > button {
        border-radius: 999px;
        background: #f59e0b;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_chatbot() -> FAQChatbot:
    faq_path = Path(__file__).resolve().parent / "faq_chatbot" / "data" / "faq.csv"
    return FAQChatbot(faq_path=faq_path)


def main() -> None:
    st.title("🤖 FAQ Chatbot")
    st.caption("University FAQ assistant")

    st.markdown(
        """
        <div style='background: rgba(255,255,255,0.12); padding: 0.7rem 0.9rem; border-radius: 10px; margin-bottom: 0.7rem; border: 1px solid rgba(255,255,255,0.18);'>
            Ask about admissions, registration, fees, exams, graduation, library, hostel, or campus services.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Ask me anything about university services and FAQ topics.",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        chatbot = load_chatbot()
        answer, match, score = chatbot.get_answer(user_input)
        bot_message = answer or "Sorry, I couldn't find a suitable answer."
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
        with st.chat_message("assistant"):
            st.write(bot_message)

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
