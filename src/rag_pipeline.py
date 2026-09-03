"""
rag_pipeline.py
---------------

RAG pipeline for The Star's Johor State Election articles.

Pipeline:

    Articles
        ↓
    Chunking
        ↓
    Dense Embeddings
        ↓
    FAISS Vector Search
        ↓
    Guardrails
        ↓
    Gemini
        ↓
    Grounded Answer + Sources

Requirements:
    pip install sentence-transformers faiss-cpu google-genai python-dotenv
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import faiss
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Load variables from .env
load_dotenv()


# Project paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "articles.json"
)


# Dense embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Gemini model comes from .env
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


# Chunk configuration
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30


# Number of chunks retrieved for each question
TOP_K = 5


# Minimum semantic similarity score
MIN_RETRIEVAL_SCORE = 0.30


# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class Chunk:
    """
    One chunk from a news article.

    Article metadata is preserved for source attribution.
    """

    chunk_id: str

    article_id: int

    title: str

    authors: List[str]

    category: str

    publish_date: str

    url: str

    text: str


# ============================================================================
# RAG RESPONSE
# ============================================================================

@dataclass
class RAGResponse:
    """
    Final response returned by the RAG pipeline.
    """

    query: str

    in_scope: bool

    answer: str

    sources: List[Dict] = field(
        default_factory=list
    )

    guardrail_reason: str = ""


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean article text before chunking.
    """

    if not text:
        return ""

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive newlines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_articles(
    articles: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Chunk]:
    """
    Split articles into overlapping chunks.

    Each chunk keeps:

        - title
        - author
        - category / section
        - publication date
        - URL
        - article text

    Overlap helps preserve context between neighbouring chunks.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks: List[Chunk] = []

    for article_id, article in enumerate(
        articles,
        start=1
    ):

        content = clean_text(
            article.get(
                "content",
                ""
            )
        )

        if not content:
            continue

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        title = (
            article.get("title")
            or "Unknown title"
        )

        authors = (
            article.get("authors")
            or []
        )

        if isinstance(
            authors,
            str
        ):
            authors = [authors]

        category = (
            article.get("category")
            or "Unknown"
        )

        publish_date = (
            article.get("date_published")
            or "Unknown"
        )

        url = (
            article.get("url")
            or ""
        )

        # ------------------------------------------------------------
        # Split article into words
        # ------------------------------------------------------------

        words = content.split()

        if not words:
            continue

        start = 0
        chunk_index = 0

        # ------------------------------------------------------------
        # Create overlapping chunks
        # ------------------------------------------------------------

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text:

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"article_{article_id}"
                            f"_chunk_{chunk_index}"
                        ),
                        article_id=article_id,
                        title=title,
                        authors=authors,
                        category=category,
                        publish_date=publish_date,
                        url=url,
                        text=chunk_text
                    )
                )

            # End of article
            if end >= len(words):
                break

            # Keep overlapping words
            start = end - overlap

            chunk_index += 1

    return chunks


# ============================================================================
# DENSE EMBEDDING MODEL
# ============================================================================

class EmbeddingModel:
    """
    Sentence Transformer used to generate dense embeddings.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL
    ):

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "Embedding model loaded."
        )

    def encode(
        self,
        texts: List[str]
    ):
        """
        Convert text into normalized dense vectors.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )


# ============================================================================
# FAISS RETRIEVER
# ============================================================================

class Retriever:
    """
    Dense vector retriever using FAISS.

    Because embeddings are normalized, inner product
    is equivalent to cosine similarity.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        embedding_model: EmbeddingModel
    ):

        if not chunks:
            raise ValueError(
                "No chunks available for retrieval."
            )

        self.chunks = chunks

        self.embedding_model = (
            embedding_model
        )

        # ------------------------------------------------------------
        # Generate dense embeddings
        # ------------------------------------------------------------

        print(
            "Generating embeddings for article chunks..."
        )

        texts = [
            f"Title: {chunk.title}\n"
            f"Date: {chunk.publish_date}\n"
            f"Section: {chunk.category}\n"
            f"Article: {chunk.text}"
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_model.encode(
                texts
            )
        )

        # ------------------------------------------------------------
        # Build FAISS index
        # ------------------------------------------------------------

        dimension = embeddings.shape[1]

        print(
            "Embedding dimension:",
            dimension
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

        print(
            "FAISS index created."
        )

        print(
            "Vectors indexed:",
            self.index.ntotal
        )

    def search(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Dict]:
        """
        Search for the most semantically relevant chunks.
        """

        if not query.strip():
            return []

        # ------------------------------------------------------------
        # Embed user query
        # ------------------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )

        # ------------------------------------------------------------
        # FAISS similarity search
        # ------------------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(score)
                }
            )

        return results


# ============================================================================
# GUARDRAILS
# ============================================================================

# Terms related to the intended topic
ELECTION_KEYWORDS = [
    "johor",
    "election",
    "elections",
    "poll",
    "polls",
    "polling",
    "vote",
    "voter",
    "voters",
    "candidate",
    "candidates",
    "seat",
    "seats",
    "constituency",
    "constituencies",
    "campaign",
    "manifesto",
    "nomination",
    "ballot",
    "turnout",
    "coalition",
    "barisan",
    "bn",
    "pakatan",
    "ph",
    "perikatan",
    "pn",
    "bersatu",
    "pas",
    "dap",
    "muda",
    "assemblyman",
    "state government",
    "menteri besar"
]


# Other Malaysian states.
# Explicit questions about these states should be rejected.
OTHER_STATES = [
    "selangor",
    "penang",
    "perak",
    "kedah",
    "kelantan",
    "terengganu",
    "pahang",
    "labuan",
    "kuala lumpur",
    "negeri sembilan",
    "melaka",
    "perlis",
    "sabah",
    "sarawak"
]


# Clearly unrelated topics
OUT_OF_SCOPE_KEYWORDS = [
    "stock",
    "stocks",
    "bursa",
    "share price",
    "share prices",
    "weather",
    "forecast",
    "food",
    "football",
    "world cup",
    "fifa",
    "actress",
    "actor",
    "songs",
    "song",
    "soccer",
    "sports",
    "recipe",
    "recipes",
    "movie",
    "movies",
    "celebrity"
]


@dataclass
class GuardrailResult:
    """
    Result of the guardrail check.
    """

    in_scope: bool

    reason: str

    relevance_score: float


def check_scope(
    query: str,
    retriever: Retriever,
    min_retrieval_score: float = MIN_RETRIEVAL_SCORE
) -> GuardrailResult:
    """
    Determine whether the question can be answered
    from the Johor election article collection.
    """

    query_lower = query.lower()

    # ------------------------------------------------------------
    # 1. Reject clearly unrelated topics
    # ------------------------------------------------------------

    for keyword in OUT_OF_SCOPE_KEYWORDS:

        if keyword in query_lower:

            return GuardrailResult(
                in_scope=False,
                reason=(
                    f"Out-of-scope topic detected: "
                    f"{keyword}"
                ),
                relevance_score=0.0
            )

    # ------------------------------------------------------------
    # 2. Reject explicit questions about other states
    # ------------------------------------------------------------

    for state in OTHER_STATES:

        if state in query_lower:

            return GuardrailResult(
                in_scope=False,
                reason=(
                    f"The question refers to "
                    f"{state.title()}, while this "
                    f"system only covers the Johor "
                    f"state election."
                ),
                relevance_score=0.0
            )

    # ------------------------------------------------------------
    # 3. Check semantic relevance
    # ------------------------------------------------------------

    results = retriever.search(
        query,
        top_k=1
    )

    top_score = (
        results[0]["score"]
        if results
        else 0.0
    )

    # ------------------------------------------------------------
    # 4. Check election-related keywords
    # ------------------------------------------------------------

    keyword_hits = [
        keyword
        for keyword in ELECTION_KEYWORDS
        if keyword in query_lower
    ]

    # ------------------------------------------------------------
    # 5. Accept if semantically relevant
    # ------------------------------------------------------------

    if top_score >= min_retrieval_score:

        return GuardrailResult(
            in_scope=True,
            reason=(
                "Question passed the semantic "
                "relevance check."
            ),
            relevance_score=top_score
        )

    # ------------------------------------------------------------
    # 6. Reject if not relevant enough
    # ------------------------------------------------------------

    if keyword_hits:

        reason = (
            "The question contains election-related "
            "terms, but the available articles do not "
            "provide sufficiently relevant context."
        )

    else:

        reason = (
            "The question does not appear to be "
            "related to the Johor state election."
        )

    return GuardrailResult(
        in_scope=False,
        reason=reason,
        relevance_score=top_score
    )


# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

def format_context(
    results: List[Dict]
) -> str:
    """
    Convert retrieved chunks into a structured
    context for Gemini.
    """

    blocks = []

    for number, result in enumerate(
        results,
        start=1
    ):

        chunk: Chunk = result["chunk"]

        if chunk.authors:

            authors = ", ".join(
                chunk.authors
            )

        else:

            authors = "Not available"

        blocks.append(
            f"""
[Source {number}]

Title:
{chunk.title}

Publication date:
{chunk.publish_date}

Author:
{authors}

Section:
{chunk.category}

URL:
{chunk.url}

Article excerpt:
{chunk.text}
"""
        )

    return "\n".join(
        blocks
    )


# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================

SYSTEM_PROMPT = """
You are a factual question-answering assistant for
The Star's coverage of the Johor state election.

Your answers must be grounded ONLY in the article excerpts
provided to you.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts, names, numbers, dates, election
   results or candidate information.

3. If the retrieved excerpts do not contain enough
   information to answer the question, say so clearly.

4. Give a concise and factual answer.

5. Remain politically neutral.

6. When making factual claims, cite the relevant source
   using the format [Source 1], [Source 2], etc.

7. Only use source numbers that actually appear in the
   provided context.

8. Do not create a separate Sources section. The application
   will display the source metadata separately.

9. Do not answer questions about other Malaysian state
   elections unless the provided context explicitly and
   sufficiently supports the answer. This system is
   specifically intended for the Johor state election.
"""


def get_gemini_client():
    """
    Create the Gemini client using the API key in .env.
    """

    api_key = (
        os.environ.get(
            "GOOGLE_API_KEY"
        )
        or
        os.environ.get(
            "GEMINI_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(
            "Gemini API key not found. "
            "Please check your .env file."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================================
# GEMINI GENERATION
# ============================================================================

def generate_answer_gemini(
    query: str,
    results: List[Dict]
) -> str:
    """
    Generate a grounded answer using Gemini.
    """

    client = get_gemini_client()

    context = format_context(
        results
    )

    prompt = f"""
Retrieved article context:

{context}

Reader question:

{query}

Answer the question using ONLY the retrieved context.

Remember:
- Do not use outside knowledge.
- Do not guess.
- Cite factual claims using [Source N].
- If the context is insufficient, clearly say that
  the available articles do not provide enough information.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=600
        )
    )

    answer = (
        response.text
        if response.text
        else ""
    )

    if not answer.strip():

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()


# ============================================================================
# SOURCE PREPARATION
# ============================================================================

def prepare_sources(
    results: List[Dict]
) -> List[Dict]:
    """
    Create a unique source list from retrieved chunks.

    Multiple chunks can come from the same article,
    so duplicate article URLs are removed.
    """

    sources = []

    seen_urls = set()

    for result in results:

        chunk: Chunk = result["chunk"]

        if chunk.url in seen_urls:
            continue

        seen_urls.add(
            chunk.url
        )

        sources.append(
            {
                "title": chunk.title,
                "date": chunk.publish_date,
                "author": (
                    ", ".join(chunk.authors)
                    if chunk.authors
                    else "Not available"
                ),
                "section": chunk.category,
                "url": chunk.url,
                "score": round(
                    result["score"],
                    4
                )
            }
        )

    return sources


# ============================================================================
# MAIN RAG FUNCTION
# ============================================================================

def answer_question(
    query: str,
    retriever: Retriever,
    top_k: int = TOP_K
) -> RAGResponse:
    """
    Complete RAG workflow.

    1. Guardrail
    2. Retrieve relevant chunks
    3. Generate grounded Gemini answer
    4. Return answer and sources
    """

    # ------------------------------------------------------------
    # Simple / greeting conversation
    # ------------------------------------------------------------

    conversational_response = handle_simple_conversation(
        query
    )

    if conversational_response:

        return RAGResponse(
            query=query,
            in_scope=False,
            answer=conversational_response
        )


    # ------------------------------------------------------------
    # Guardrail
    # ------------------------------------------------------------

    guardrail = check_scope(
        query,
        retriever
    )

    if not guardrail.in_scope:

        return RAGResponse(
            query=query,
            in_scope=False,
            answer=(
                "I'm sorry, but I can only answer "
                "questions about The Star's coverage "
                "of the Johor state election."
            ),
            guardrail_reason=(
                guardrail.reason
            )
        )

    # ------------------------------------------------------------
    # Retrieve relevant chunks
    # ------------------------------------------------------------

    results = retriever.search(
        query,
        top_k=top_k
    )

    # ------------------------------------------------------------
    # No relevant results
    # ------------------------------------------------------------

    if not results:

        return RAGResponse(
            query=query,
            in_scope=True,
            answer=(
                "I could not find relevant information "
                "in the available The Star articles."
            ),
            guardrail_reason=(
                guardrail.reason
            )
        )

    # ------------------------------------------------------------
    # Check retrieval quality
    # ------------------------------------------------------------

    relevant_results = [
        result
        for result in results
        if result["score"] >= MIN_RETRIEVAL_SCORE
    ]

    if not relevant_results:

        return RAGResponse(
            query=query,
            in_scope=False,
            answer=(
                "I could not find enough relevant "
                "information in the available articles "
                "to answer this question reliably."
            ),
            guardrail_reason=(
                "Retrieved articles did not meet "
                "the minimum relevance threshold."
            )
        )

    # ------------------------------------------------------------
    # Generate Gemini answer
    # ------------------------------------------------------------

    try:

        answer = generate_answer_gemini(
            query,
            relevant_results
        )

    except Exception as error:

        print(
            "\nGemini generation error:"
        )

        print(
            error
        )

        return RAGResponse(
            query=query,
            in_scope=True,
            answer=(
                "I found relevant articles, but I "
                "could not generate an answer right now. "
                "Please check the Gemini API configuration."
            ),
            sources=prepare_sources(
                relevant_results
            ),
            guardrail_reason=(
                f"Gemini generation failed: {error}"
            )
        )

    # ------------------------------------------------------------
    # Prepare sources
    # ------------------------------------------------------------

    sources = prepare_sources(
        relevant_results
    )

    return RAGResponse(
        query=query,
        in_scope=True,
        answer=answer,
        sources=sources,
        guardrail_reason=(
            guardrail.reason
        )
    )


# ============================================================================
# DEBUG / EVALUATION
# ============================================================================

def display_retrieval_results(
    results: List[Dict]
):
    """
    Display retrieved chunks for debugging and evaluation.
    """

    print("\n")
    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    if not results:

        print(
            "No results."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk: Chunk = result["chunk"]

        print("\n")
        print("-" * 80)

        print(
            f"RESULT #{rank}"
        )

        print("-" * 80)

        print(
            "Chunk ID:",
            chunk.chunk_id
        )

        print(
            "Article:",
            chunk.title
        )

        print(
            "Date:",
            chunk.publish_date
        )

        print(
            "Author:",
            (
                ", ".join(chunk.authors)
                if chunk.authors
                else "Not available"
            )
        )

        print(
            "Section:",
            chunk.category
        )

        print(
            "Similarity:",
            round(
                result["score"],
                4
            )
        )

        print(
            "URL:",
            chunk.url
        )

        print("\nExcerpt:")

        print(
            chunk.text[:700]
        )


# ============================================================================
# TEST QUERY
# ============================================================================

def run_test_query(
    query: str,
    retriever: Retriever
):
    """
    Run one question through the complete RAG pipeline.
    """

    print("\n")
    print("=" * 80)
    print("TEST QUESTION")
    print("=" * 80)

    print(
        query
    )

    # ------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------

    results = retriever.search(
        query,
        top_k=TOP_K
    )

    display_retrieval_results(
        results
    )

    # ------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------

    response = answer_question(
        query,
        retriever
    )

    print("\n")
    print("=" * 80)
    print("RAG ANSWER")
    print("=" * 80)

    print(
        response.answer
    )

    # ------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------

    if response.sources:

        print("\n")
        print("=" * 80)
        print("SOURCES")
        print("=" * 80)

        for number, source in enumerate(
            response.sources,
            start=1
        ):

            print(
                f"\nSource {number}"
            )

            print(
                "Title:",
                source["title"]
            )

            print(
                "Date:",
                source["date"]
            )

            print(
                "Author:",
                source["author"]
            )

            print(
                "Section:",
                source["section"]
            )

            print(
                "Similarity:",
                source["score"]
            )

            print(
                "URL:",
                source["url"]
            )

    print("\n")
    print(
        "Guardrail:",
        response.guardrail_reason
    )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_articles(
    path: str = DATA_PATH
) -> List[Dict]:
    """
    Load scraped articles from JSON.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        articles = json.load(
            file
        )

    if not isinstance(
        articles,
        list
    ):

        raise ValueError(
            "articles.json must contain a list of articles."
        )

    return articles

# ============================================================================
# SIMPLE / GREETING CONVERSATION HANDLER
# ============================================================================

def handle_simple_conversation(
    query: str
) -> str | None:
    """
    Handle simple greetings and conversational messages
    before sending the question through the RAG guardrail.

    Returns:
        A friendly response if the message is conversational.
        None if the message should continue to the RAG pipeline.
    """

    normalized_query = re.sub(
        r"\s+",
        " ",
        query.strip().lower()
    )

    # Simple greetings
    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "olla"
    }

    if normalized_query in greetings:

        return (
            "👋 Hello! You can ask me questions about "
            "The Star's coverage of the Johor state election."
        )

    # Simple help requests
    help_requests = {
        "can you help me",
        "can you help me?",
        "help me",
        "help",
        "help me please",
        "can you help",
        "can you help?"
    }

    if normalized_query in help_requests:

        return (
            "Sure! 😊 Ask me a question about The Star's "
            "coverage of the Johor state election."
        )

    # Simple test messages
    test_messages = {
        "test",
        "testing",
        "test test"
    }

    if normalized_query in test_messages:

        return (
            "👋 I'm ready! Try asking me a question about "
            "The Johor state election."
        )

    return None

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("JOHOR ELECTION RAG PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # STEP 1 — LOAD ARTICLES
    # ------------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("STEP 1: LOAD ARTICLES")
    print("-" * 80)

    articles = load_articles()

    print(
        "Dataset:",
        DATA_PATH
    )

    print(
        "Articles loaded:",
        len(articles)
    )

    # ------------------------------------------------------------------------
    # STEP 2 — CHUNK ARTICLES
    # ------------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("STEP 2: CHUNK ARTICLES")
    print("-" * 80)

    chunks = chunk_articles(
        articles
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    if chunks:

        chunk_lengths = [
            len(chunk.text.split())
            for chunk in chunks
        ]

        print(
            "Minimum chunk words:",
            min(chunk_lengths)
        )

        print(
            "Maximum chunk words:",
            max(chunk_lengths)
        )

        print(
            "Average chunk words:",
            round(
                sum(chunk_lengths)
                / len(chunk_lengths),
                2
            )
        )

    # ------------------------------------------------------------------------
    # STEP 3 — LOAD EMBEDDING MODEL
    # ------------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("STEP 3: LOAD DENSE EMBEDDING MODEL")
    print("-" * 80)

    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    # ------------------------------------------------------------------------
    # STEP 4 — BUILD VECTOR INDEX
    # ------------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("STEP 4: BUILD FAISS VECTOR INDEX")
    print("-" * 80)

    retriever = Retriever(
        chunks,
        embedding_model
    )

    print(
        "Retriever ready."
    )

    # ------------------------------------------------------------------------
    # STEP 5 — TEST IN-SCOPE QUESTION
    # ------------------------------------------------------------------------

    test_query = (
        "Who won the Pulai seat in the Johor state election?"
    )

    run_test_query(
        test_query,
        retriever
    )

    # ------------------------------------------------------------------------
    # STEP 6 — TEST OUT-OF-SCOPE QUESTION
    # ------------------------------------------------------------------------

    out_of_scope_query = (
        "Who won the Selangor state election?"
    )

    run_test_query(
        out_of_scope_query,
        retriever
    )

    print("\n")
    print("=" * 80)
    print("RAG PIPELINE TEST COMPLETE")
    print("=" * 80)