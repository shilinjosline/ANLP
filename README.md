# Simple Retrieval-Augmented Generation (RAG) System using Ollama

This project implements a minimal Retrieval-Augmented Generation (RAG) pipeline in Python using local inference with Ollama. It evaluates how changing the number of retrieved context chunks (`top_n`) impacts the quality of generated answers from a local language model.

## What the Code Does

The `demo.py` script performs the following tasks:

1. Loads a dataset from `cat-facts.txt`, treating each line as a chunk of factual knowledge.
2. Embeds each chunk using `ollama.embed()` and stores them in a simple vector database.
3. Implements cosine similarity to retrieve the most relevant chunks for a given user query.
4. Tests multiple retrieval settings (`top_n` from 1 to 5) across a small set of example queries.
5. Generates responses using a local instruction-tuned LLM with the retrieved context.
6. Logs all outputs in `rag_eval_log.txt` for review.

## How to Run

  ### 1. Prerequisites
  - Python 3.7+
  - Ollama installed and running
  - Required models:
    ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
    ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  - Python library:
    pip install ollama
  
  ### 2. Run the Code
  Make sure `cat-facts.txt` is located at the specified path in the script (or update it), then run:
    python demo.py
  
  ### 3. Output
  All chatbot responses across different `top_n` settings will be saved in:
    rag_eval_log.txt

## Reflection

### What limitations did you observe?

- Response verbosity & hallucination: At higher `top_n` values, the LLM produced longer, sometimes repetitive or incorrect answers due to less relevant context being included.
- Limited coverage: Since the dataset (`cat-facts.txt`) is short and domain-specific, the system can only answer a narrow range of questions.
- Basic retrieval: Cosine similarity over raw embeddings works, but lacks nuance (e.g., no semantic reranking or disambiguation).

### What could be improved with a larger dataset or better models?

- More diverse dataset: A larger and more varied knowledge base (e.g., Wikipedia, domain articles) would enable the system to answer broader and more complex queries.
- Advanced embedding models: More powerful or domain-adapted embedding models could enhance retrieval quality.
- Larger instruction-tuned LLMs: Models like LLaMA 3 8B or Mistral-Instruct would generate more fluent, factual, and detailed responses.
- Reranking and filtering: Incorporating a second stage reranker or using hybrid retrieval (BM25 + embeddings) could further refine results.
