"""
app.py
------
Streamlit chat UI for the Johor Election RAG prototype.

Run with:
    streamlit run app.py
"""

import os
import sys

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
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header("🗳️ Johor Election RAG")

    st.markdown(
        """
        **Data source**

        The Star Malaysia

        **Coverage**

        Johor state election

        **Retrieval**

        FAISS + dense embeddings

        **Embedding model**

        `all-MiniLM-L6-v2`

        **Generation**

        Gemini
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================================
# MAIN TITLE
# ============================================================================

st.title(
    "🗳️ Johor State Election — Ask The Star"
)

st.caption(
    "Ask questions about The Star's coverage of the "
    "Johor state election. Answers are grounded only "
    "in the retrieved articles."
)


# ============================================================================
# LOAD RAG RETRIEVER
# ============================================================================

@st.cache_resource
def load_retriever():

    # ------------------------------------------------------------
    # Check dataset
    # ------------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # ------------------------------------------------------------
    # Load articles
    # ------------------------------------------------------------

    articles = load_articles()

    # ------------------------------------------------------------
    # Chunk articles
    # ------------------------------------------------------------

    chunks = chunk_articles(
        articles
    )

    # ------------------------------------------------------------
    # Load embedding model
    # ------------------------------------------------------------

    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    # ------------------------------------------------------------
    # Build FAISS retriever
    # ------------------------------------------------------------

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
# CHAT HISTORY
# ============================================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================================
# WELCOME MESSAGE
# ============================================================================

if len(st.session_state.history) == 0:

    st.info(
        "💡 Example questions:\n\n"
        "- How many candidates did Pakatan announce?\n"
        "- How many seats did BN win?\n"
        "- When was polling day?\n"
        "- Who won the Pulai seat?"
    )


# ============================================================================
# DISPLAY PREVIOUS CHAT
# ============================================================================

for turn in st.session_state.history:

    with st.chat_message(
        turn["role"]
    ):

        st.markdown(
            turn["content"]
        )

        # --------------------------------------------------------
        # Display sources for assistant messages
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
    # Display user question
    # ------------------------------------------------------------

    st.session_state.history.append(
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

                st.session_state.history.append(
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

                # Keep technical error available during development
                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )