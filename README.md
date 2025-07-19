# Simple Retrieval-Augmented Generation (RAG) System using Ollama

This project implements a minimal Retrieval-Augmented Generation (RAG) pipeline in Python using local inference with [Ollama](https://ollama.com). It evaluates how changing the number of retrieved context chunks (`top_n`) impacts the quality of generated answers from a locally hosted language model.

---

## What the Code Does

The `demo.py` script performs the following tasks:

1. **Loads** a dataset from `cat-facts.txt`, treating each line as a chunk of factual knowledge.
2. **Embeds** each chunk using `ollama.embed()` and stores them in a vector database (`VECTOR_DB`).
3. **Implements** cosine similarity to retrieve the most relevant chunks for a given user query.
4. **Tests** multiple retrieval settings (`top_n` from 1 to 5) across a small set of evaluation queries.
5. **Generates** responses using a local instruction-tuned LLM with the retrieved context.
6. **Logs** all outputs in `rag_eval_log.txt` for analysis.

---

## How to Run

### 1. Prerequisites

- Python 3.7+
- Ollama installed and running: [https://ollama.com](https://ollama.com)
- Required models:
  ```bash
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

- More diverse dataset: A larger and more varied knowledge base would enable the system to answer broader and more complex queries.
- Advanced embedding models: More powerful or domain-adapted embedding models could enhance retrieval quality.
- Larger instruction-tuned LLMs: Bigger models would generate more fluent, factual, and detailed responses.
- Reranking and filtering: Incorporating a second stage reranker or using hybrid retrieval could further refine results.

## Reflection on Experimentation Tasks

### 1. Trying Different `top_n` Values in Retrieval

In the evaluation, we tested values of `top_n` ranging from 1 to 5 to observe how the number of retrieved context chunks affects the quality of the chatbot’s response.  
We found that:

- **Lower values** (e.g., `top_n=1`) often led to concise and highly relevant answers, but occasionally lacked depth or completeness.
- **Higher values** (e.g., `top_n=4` or `5`) included more information, but sometimes introduced off-topic or redundant context, which slightly reduced answer accuracy or clarity.

This confirmed the **trade-off between precision and recall** in retrieval: fewer chunks improve focus, more chunks improve breadth — but at the cost of possible noise.

---

### 2. Modifying the Prompt Template

We implemented a clear, custom system prompt instructing the LLM to answer only based on retrieved context and avoid fabricating facts. This constraint worked well in most cases, encouraging the model to remain grounded in retrieved data. However, prompt quality still influenced the outcome — subtle changes in wording could make the model more cautious or more verbose.

**Future improvements could include:**

- More explicit fallback instructions (e.g., “If no context is relevant, say ‘I don’t know.’”)
- Reranking retrieved context based on prompt relevance

---

### 3. Asking Questions with Varying Specificity

Our evaluation queries were intentionally diverse:

- Some were **factual** (“What is the average lifespan of a cat?”),
- Others were **behavioral** (“Why do cats purr?”), and
- Some involved **common myths** or **biological mechanisms**.

This range revealed that:

- Simple factual queries were answered well even with just one or two chunks.
- More nuanced or explanatory questions benefited from higher `top_n` values, as they required piecing together multiple facts.
- Specificity helped guide retrieval better — vague queries often returned broader, less relevant facts.

This underscores the importance of **query clarity** in real-world RAG systems.

