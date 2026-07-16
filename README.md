# Multi-Agent RAG with Mistral

[Русская версия](README_RU.md)

A compact educational Retrieval-Augmented Generation project built without LangChain, LangGraph, FAISS, or a vector database. It demonstrates the core mechanics directly in Python: document chunking, local embeddings, semantic search with NumPy, generation through the Mistral API, and a three-role agent pipeline.

## Architecture

```text
User question
    ↓
Router Agent
    ↓
Semantic Retriever
    ↓
RAG Agent
    ↓
Checker Agent
    ↓
Final answer
```

- **Router Agent** decides whether the question belongs to the local knowledge base.
- **Retriever** creates a question embedding and selects the most similar text chunks.
- **RAG Agent** answers using only the retrieved context.
- **Checker Agent** removes claims that are not supported by that context.

The retriever is deterministic Python code, not an LLM agent.

## Stack

- Python 3.10+
- Mistral API
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- NumPy
- python-dotenv

## Project structure

```text
rag-multiagent/
├── data/
│   └── routepilot.txt
├── src/
│   ├── __init__.py
│   ├── agents.py
│   ├── documents.py
│   ├── pipeline.py
│   └── retrieval.py
├── .env.example
├── .gitignore
├── main.py
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/mchibisov/rag-multiagent.git
cd rag-multiagent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Create the local environment file:

```bash
cp .env.example .env
```

Then replace `your_mistral_api_key` in `.env` with your key. The `.env` file is ignored by Git.

## Run

```bash
python main.py
```

Example questions:

```text
How long is the trial period?
How much does the basic plan cost?
Does RoutePilot control vehicles directly?
Who is the CEO of the company?
```

Type `exit` to stop the application.

## How retrieval works

1. The document is split into paragraph-based chunks.
2. A local Sentence Transformer converts each chunk into a normalized vector.
3. The question is converted into a vector using the same model.
4. NumPy computes similarity scores using matrix multiplication.
5. The two most similar chunks are passed to Mistral as context.

Because vectors are normalized, the dot product is equivalent to cosine similarity.

## Current limitations

- One local text document is indexed in memory on every launch.
- Search is a linear NumPy scan, suitable only for a small educational corpus.
- Router and checker decisions are probabilistic LLM outputs.
- Prompt-injection resistance is limited and not a security guarantee.
- There is no persistent vector index, web interface, or automated evaluation suite.

## Why this project is intentionally simple

The goal is to expose every important RAG step without hiding it behind a framework. A production system would usually add persistent indexing, observability, evaluation, stronger input controls, retries, model configuration, and scalable retrieval.
