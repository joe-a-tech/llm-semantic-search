#!/usr/bin/env python3

import sys
import math
import requests
import tiktoken

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
ENCODING = "cl100k_base"

DOCUMENTS = [
    "Docker is a platform for building and running containers.",
    "Kubernetes is used to orchestrate containerized applications.",
    "Linux uses the kernel to manage hardware and processes.",
    "Embeddings turn text into vectors that capture semantic meaning.",
    "Tokenisation splits text into chunks that models can process.",
    "Cosine similarity measures how close two vectors are in direction.",
    "RAG combines retrieval with a language model to answer questions.",
    "Python is commonly used for AI tooling and experimentation.",
]


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize_text(text):
    enc = tiktoken.get_encoding(ENCODING)
    token_ids = enc.encode(text)
    token_chunks = [enc.decode([t]) for t in token_ids]
    return token_ids, token_chunks


def build_index(documents):
    index = []
    for doc in documents:
        emb = get_embedding(doc)
        index.append((doc, emb))
    return index


def search(query, index, top_k=3):
    print("\n=== STEP 1: TOKENISE QUERY ===")
    token_ids, token_chunks = tokenize_text(query)
    print(f"Query: {query}")
    print(f"Token count: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print("Token chunks:")
    for i, chunk in enumerate(token_chunks, start=1):
        print(f"  {i:02d}: {chunk!r}")

    print("\n=== STEP 2: INITIAL TOKEN VECTORS (internal) ===")
    print(f"{len(token_ids)} token IDs -> {len(token_ids)} internal token vectors")
    print("(not exposed by this API)")
    print("These are then updated by transformer attention")
    print("and pooled into one final embedding.")

    print("\n=== STEP 3: EMBED QUERY ===")
    query_emb = get_embedding(query)
    print(f"Vector length: {len(query_emb)}")
    print(f"First 5 values: {[round(x, 4) for x in query_emb[:5]]}")

    print("\n=== STEP 4: COMPARE AGAINST DOCUMENTS ===")
    scored = []

    for i, (doc, emb) in enumerate(index, start=1):
        score = cosine_similarity(query_emb, emb)
        print(f"\nDoc {i}: {doc}")
        print(f"Similarity: {round(score, 4)}")
        scored.append((score, doc))

    print("\n=== STEP 5: SORT RESULTS ===")
    scored.sort(reverse=True, key=lambda x: x[0])

    for i, (score, doc) in enumerate(scored, start=1):
        print(f"{i}. [{round(score, 4)}] {doc}")

    return scored[:top_k]


def main():
    if len(sys.argv) < 2:
        print('Usage: python semantic_search.py "your query here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    try:
        index = build_index(DOCUMENTS)
        search(query, index)

    except requests.exceptions.RequestException as exc:
        print(f"Error talking to Ollama: {exc}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
