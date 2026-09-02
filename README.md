# Johor Election RAG Chatbot — Section B Prototype

A Retrieval-Augmented Generation (RAG) prototype that allows users to ask
questions about **The Star's coverage of the Johor state election** and
receive grounded answers with source citations.

The system uses real The Star articles, dense vector embeddings,
FAISS similarity search, Gemini and Streamlit. It also includes
guardrails to politely decline out-of-scope questions.

## Project Structure

```text
rag_johor_election/
│
├── README.md
├── requirements.txt
├── architecture_design.md
├── app.py
│
├── data/
│   └── articles.json
│
└── src/
    ├── collect_urls.py
    ├── scraper.py
    ├── scrape_all.py
    ├── validate_dataset.py
    └── rag_pipeline.py
```

## Requirements

- Python 3.12+
- Internet connection for scraping articles
- Gemini API key

## Local Setup

### 1. Clone the repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd rag_johor_election
```

### 2. Create and activate virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model
```

Alternatively, `GEMINI_API_KEY` can be used instead of
`GOOGLE_API_KEY`.

**Insert your own 'LLM_API_KEY' and 'LLM_MODEL'**

### 5. Run the application

The repository already contains the scraped dataset in:

```text
data/articles.json
```

Run:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Refreshing the Dataset

To collect and scrape a new set of articles:

```bash
python src/collect_urls.py
python src/scrape_all.py
```

The scraped data is saved to:

```text
data/articles.json
```

Validate the dataset with:

```bash
python src/validate_dataset.py
```

## Example Questions

**In-scope:**

> How many candidates did BN announce for the Johor polls?

> When was the Johor election?

**Out-of-scope:**

> Who won the Selangor state election?

> What is the weather in Johor?

The system should politely decline questions outside the Johor election
scope.

## Technology Stack

- **Python** — application development
- **Playwright** — dynamic search result collection
- **Requests / BeautifulSoup** — article extraction
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — dense embeddings
- **FAISS** — vector similarity search
- **Gemini** — grounded answer generation
- **Streamlit** — interactive chatbot interface

## System Design

See [`architecture_design.md`](architecture_design.md) for the end-to-end
architecture, Mermaid system design diagram, component justification,
retrieval flow and guardrail pathing.