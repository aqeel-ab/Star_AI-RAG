"""
app.py
------
Streamlit chat UI for the Johor Election RAG prototype.

Run with:
    streamlit run app.py / python -m streamlit run app.py
"""

import os
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv


# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

load_dotenv()


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "articles.json"
)

sys.path.insert(
    0,
    SRC_PATH
)


# ============================================================================
# IMPORT RAG PIPELINE
# ============================================================================

from rag_pipeline import (  # noqa: E402
    load_articles,
    chunk_articles,
    EmbeddingModel,
    Retriever,
    answer_question,
    EMBEDDING_MODEL
)


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Johor Election Q&A — The Star",
    page_icon="🗳️",
    layout="wide"
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


def create_new_chat():
    """
    Create a new empty chat session.
    """

    chat_id = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "history": []
    }

    st.session_state.current_chat_id = chat_id


def get_current_chat():
    """
    Return the currently selected chat.
    """

    chat_id = st.session_state.current_chat_id

    if (
        chat_id is None
        or chat_id not in st.session_state.chats
    ):
        create_new_chat()

    return st.session_state.chats[
        st.session_state.current_chat_id
    ]


# Create first chat
get_current_chat()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header("🗳️ Johor Election RAG")

    # ------------------------------------------------------------
    # New Chat
    # ------------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        create_new_chat()
        st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # Chat History
    # ------------------------------------------------------------

    st.subheader("Chat History")

    chats = st.session_state.chats

    # Show newest chats first
    chat_items = list(
        chats.items()
    )[::-1]

    has_previous_chats = False

    for chat_id, chat in chat_items:
        # Only show chats that contain messages
        if not chat["history"]:
            continue

        has_previous_chats = True

        title = chat["title"]

        if len(title) > 32:
            title = title[:32] + "..."

        if st.button(
            f"💬 {title}",
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat_id = chat_id
            st.rerun()


    if not has_previous_chats:
        st.caption(
            "No previous chats yet."
        )

    st.divider()

    # ------------------------------------------------------------
    # Clear All Chats
    # ------------------------------------------------------------

    if st.button(
        "🗑️ Clear All Chats",
        use_container_width=True
    ):

        st.session_state.chats = {}

        create_new_chat()

        st.rerun()


# ============================================================================
# CURRENT CHAT
# ============================================================================

current_chat = get_current_chat()

history = current_chat["history"]


# ============================================================================
# MAIN TITLE
# ============================================================================

st.title(
    "🗳️ Johor State Election — Ask The Star"
)

st.caption(
    "Ask questions about The Star's coverage of the "
    "Johor state election. Answers are generated from "
    "retrieved articles and include source citations."
)


# ============================================================================
# LOAD RAG RETRIEVER
# ============================================================================

@st.cache_resource
def load_retriever():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    articles = load_articles()

    chunks = chunk_articles(
        articles
    )

    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    retriever = Retriever(
        chunks,
        embedding_model
    )

    return retriever


# ============================================================================
# INITIALIZE RETRIEVER
# ============================================================================

try:

    with st.spinner(
        "Loading articles and building the vector index..."
    ):

        retriever = load_retriever()

except Exception as error:

    st.error(
        "Unable to load the RAG system."
    )

    st.exception(error)

    st.stop()


# ============================================================================
# GEMINI API KEY CHECK
# ============================================================================

if not (
    os.environ.get("GOOGLE_API_KEY")
    or
    os.environ.get("GEMINI_API_KEY")
):

    st.warning(
        "Gemini API key was not found. "
        "Please check your .env file."
    )


# ============================================================================
# WELCOME MESSAGE
# ============================================================================

if len(history) == 0:

    st.info(
        "💡 Example questions:\n\n"
        "- How many candidates did Pakatan announce "
        "for the Johor polls?\n"
        "- How many seats did BN win in the Johor polls?\n"
        "- When was polling day for the Johor election?\n"
        "- Who won the Pulai seat?"
    )


# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================

for turn in history:

    with st.chat_message(
        turn["role"]
    ):

        st.markdown(
            turn["content"]
        )

        # --------------------------------------------------------
        # Display sources
        # --------------------------------------------------------

        if (
            turn["role"] == "assistant"
            and turn.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for number, source in enumerate(
                    turn["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**{number}. {source['title']}**"
                    )

                    st.caption(
                        f"Published: {source['date']}"
                    )

                    st.link_button(
                        "Read article",
                        source["url"]
                    )


# ============================================================================
# CHAT INPUT
# ============================================================================

query = st.chat_input(
    "Ask about the Johor state election..."
)


# ============================================================================
# PROCESS QUESTION
# ============================================================================

if query:

    # ------------------------------------------------------------
    # Update chat title using first question
    # ------------------------------------------------------------

    if current_chat["title"] == "New Chat":

        current_chat["title"] = query.strip()

    # ------------------------------------------------------------
    # Display user question
    # ------------------------------------------------------------

    current_chat["history"].append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # ------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching The Star articles..."
        ):

            try:

                response = answer_question(
                    query,
                    retriever
                )

                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(
                    response.answer
                )

                # ------------------------------------------------
                # Display sources
                # ------------------------------------------------

                sources = []

                if (
                    response.in_scope
                    and response.sources
                ):

                    sources = response.sources

                    with st.expander(
                        "📚 Sources"
                    ):

                        for number, source in enumerate(
                            response.sources,
                            start=1
                        ):

                            st.markdown(
                                f"**{number}. "
                                f"{source['title']}**"
                            )

                            st.caption(
                                f"Published: "
                                f"{source['date']}"
                            )

                            st.link_button(
                                "Read article",
                                source["url"]
                            )

                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                current_chat["history"].append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources
                    }
                )

            except Exception as error:

                error_message = (
                    "I couldn't generate an answer "
                    "right now. Please check the "
                    "Gemini API configuration."
                )

                st.error(
                    error_message
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)

                current_chat["history"].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )