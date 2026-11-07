# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a search step with a language model.
Before answering, the system retrieves passages relevant to the question from a
document collection and places them in the model's prompt.

RAG reduces hallucination because the model can ground its answer in real sources,
and it lets the model use private or recent information it was never trained on.
Citing the retrieved passages makes answers easy to verify.

Retrieval can be lexical, such as BM25, which scores documents by term frequency and
inverse document frequency, or dense, which compares embedding vectors. BM25 is fast,
needs no model, and is a strong baseline for keyword-heavy queries.
