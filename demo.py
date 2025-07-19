#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate RAG system with different top_n retrieval values.
"""

import ollama
import math

# === Configuration ===
EMBEDDING_MODEL = 'bge-base-en-v1.5'
LANGUAGE_MODEL = 'llama-3.2-1b-instruct'
DATA_FILE = '/Users/shilinjosline/Downloads/cat-facts.txt'
LOG_FILE = 'rag_eval_log.txt'

# === Load and Chunk the Dataset ===
dataset = []
with open(DATA_FILE, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file if line.strip()]

# === Embed the Dataset ===
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

print("Building vector database...")
for chunk in dataset:
    add_chunk_to_database(chunk)

# === Cosine Similarity Function ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    return dot / (norm_a * norm_b + 1e-8)

# === Retrieval Function ===
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === Evaluation Questions ===
evaluation_queries = [
    "What is the average lifespan of a cat?",
    "Do cats like milk?",
    "Why do cats purr?",
    "Can cats see in the dark?",
    "How do cats land on their feet?"
]

# === Evaluate Multiple top_n Values and Log Output ===
top_n_values = [1, 2, 3, 4, 5]

with open(LOG_FILE, 'w', encoding='utf-8') as log:
    for top_n in top_n_values:
        log.write(f"\n{'='*20} TOP_N = {top_n} {'='*20}\n")
        print(f"\n🔎 Testing top_n = {top_n}")

        for query in evaluation_queries:
            log.write(f"\nQuery: {query}\n")
            print(f"→ Query: {query}")

            retrieved = retrieve(query, top_n=top_n)

            log.write("Retrieved Context:\n")
            for chunk, sim in retrieved:
                log.write(f" - ({sim:.2f}) {chunk}\n")

            instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([' - ' + chunk for chunk, _ in retrieved])}
"""

            # Generate the chatbot response
            log.write("Chatbot Response:\n")
            print("Chatbot:", end=' ')
            try:
                stream = ollama.chat(
                    model=LANGUAGE_MODEL,
                    messages=[
                        {'role': 'system', 'content': instruction_prompt},
                        {'role': 'user', 'content': query},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    log.write(content)
                print()  # newline
                log.write("\n" + "-"*40 + "\n")
            except Exception as e:
                log.write(f"\nError generating response: {str(e)}\n")
                print(f"\nError: {e}")
