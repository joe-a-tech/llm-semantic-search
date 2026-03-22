# llm-semantic-search

Minimal CLI tool demonstrating semantic search using LLM embeddings and cosine similarity.

This project shows how to:
- convert text into embeddings (vectors)
- compare meaning instead of keywords
- rank results based on semantic similarity

---

## Overview

This is a simplified version of how modern AI retrieval systems work.

It follows the same core idea used in:
- vector databases
- semantic search engines
- Retrieval-Augmented Generation (RAG)

---

## How It Works

### Preparation (in this demo: done at runtime)

1. Each document is:
   - tokenised
   - converted to token IDs
   - expanded into vectors
   - processed by a transformer
   - merged into **one final embedding vector**

2. These vectors are stored in memory

---

### Query Time

3. Your query goes through the same process:
   - tokenisation
   - embedding → **one query vector**

4. The query vector is compared to each document vector

5. A similarity score is calculated

6. Results are sorted and returned

---

## Key Concept

> Everything becomes a vector → then vectors are compared

---

## Requirements

- Python 3.10+
- Ollama installed and running

Install dependencies:

    pip install -r requirements.txt

Example `requirements.txt`:

    requests
    tiktoken

---

## Setup

Start Ollama:

    ollama serve

Pull the embedding model:

    ollama pull nomic-embed-text

---

## Usage

### Run a search

    python semantic_search.py "how do containers work?"

---

## Example Output

    === STEP 1: TOKENISE QUERY ===
    Token IDs: [12840, 527, 79242, 552, 1903, 315]

    === STEP 2: INITIAL TOKEN VECTORS (internal) ===
    6 token IDs -> 6 internal token vectors

    === STEP 3: EMBED QUERY ===
    Vector length: 768

    === STEP 4: COMPARE AGAINST DOCUMENTS ===
    Doc 1: Docker is a platform for building and running containers.
    Similarity: 0.6567

    === STEP 5: SORT RESULTS ===
    1. [0.6736] Kubernetes is used to orchestrate containerized applications.
    2. [0.6567] Docker is a platform for building and running containers.

---

## Notes

- This demo recomputes document embeddings on each run
- Real systems precompute and store embeddings
- Tokenisation uses `cl100k_base` for illustration
- Internal token vectors are not exposed by the API

---

## Real-World Equivalent

| This Project        | Real System             |
|--------------------|------------------------|
| DOCUMENTS list     | database / files       |
| embeddings         | embedding service      |
| cosine similarity  | vector search engine   |
| CLI output         | API / application      |

---

## Next Steps

- Load documents from files instead of hardcoded list
- Add chunking for large documents
- Persist embeddings to disk
- Integrate with an LLM (RAG)

---

## License

MIT
