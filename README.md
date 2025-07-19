# Simple RAG System with Ollama (RAG Eval)

This project implements a **simple Retrieval-Augmented Generation (RAG)** pipeline using [Ollama](https://ollama.com) for **local language model inference**. The script:

- Loads a dataset of cat facts (`cat-facts.txt`)
- Embeds each line using a local embedding model (`bge-base-en-v1.5`)
- Stores embeddings in a vector store
- Accepts user queries
- Retrieves the top-N most relevant facts based on cosine similarity
- Generates a response using a local language model (`llama-3.2-1b-instruct`)
- Evaluates performance across multiple `top_n` values (`1` to `5`) for retrieval
- Logs outputs for qualitative analysis in `rag_eval_log.txt`

---

## How to Run

1. **Requirements**
   - Python 3.7+
   - Ollama installed and configured with the following models:
     - `bge-base-en-v1.5` (for embedding)
     - `llama-3.2-1b-instruct` (for generation)
   - File: Edit the location of the 'cat-facts.txt`

2. **Run the script**
   ```bash
   python demo.py

## Observations
What limitations did you observe?
1. Verbose / hallucinated outputs: With larger top_n values (e.g., 5), the model sometimes included redundant or unrelated information.
2. Shallow reasoning: The model occasionally gave surface-level answers, lacking deeper understanding or explanation.
3. Repetitive content: Similar chunks caused the model to repeat facts or phrases across different generations.

What could be improved with a larger dataset or more advanced models?
1. More diverse knowledge: A larger dataset would allow the system to answer more complex and varied questions.
2. Better embeddings: Using advanced embedding models could improve retrieval accuracy and relevance.
3. Stronger generation model: Upgrading to a larger instruction-tuned LLM would likely yield more fluent, accurate, and context-aware responses.
4. Reranking or filtering: Introducing a reranker model or chunk deduplication could reduce noise in retrieved context.



