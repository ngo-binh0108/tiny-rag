"""tiny-rag: a minimal retrieval-augmented generation toolkit."""

from .retriever import BM25Index, Chunk, chunk_text, load_documents, tokenize

__all__ = ["BM25Index", "Chunk", "chunk_text", "load_documents", "tokenize"]
__version__ = "0.1.0"
