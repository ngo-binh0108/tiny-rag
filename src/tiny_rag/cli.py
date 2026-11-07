"""Command-line interface: `tiny-rag index | search | ask`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .retriever import BM25Index

DEFAULT_INDEX = ".tiny_rag_index.json"


def _load(args: argparse.Namespace) -> BM25Index:
    path = Path(args.index)
    if not path.exists():
        sys.exit(f"No index at {path}. Run `tiny-rag index <docs>` first.")
    return BM25Index.load(path)


def cmd_index(args: argparse.Namespace) -> None:
    idx = BM25Index.from_path(args.docs, max_words=args.chunk_words, overlap=args.overlap)
    if not idx.chunks:
        sys.exit(f"No text documents found under {args.docs}.")
    idx.save(args.index)
    sources = len({c.source for c in idx.chunks})
    print(f"Indexed {len(idx.chunks)} chunks from {sources} file(s) -> {args.index}")


def cmd_search(args: argparse.Namespace) -> None:
    hits = _load(args).search(args.query, k=args.k)
    if not hits:
        print("No matches.")
    for rank, (chunk, score) in enumerate(hits, start=1):
        preview = chunk.text[:160] + ("..." if len(chunk.text) > 160 else "")
        print(f"{rank}. {chunk.id}  (score {score:.2f})\n   {preview}\n")


def cmd_ask(args: argparse.Namespace) -> None:
    from .generator import answer

    hits = _load(args).search(args.question, k=args.k)
    if not hits:
        sys.exit("No relevant passages found in the index for that question.")
    try:
        import anthropic
    except ImportError:
        sys.exit("The `ask` command needs the Anthropic SDK: pip install 'tiny-rag[llm]'")
    try:
        print(answer(args.question, hits, model=args.model, effort=args.effort))
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed. Set ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API. Try again shortly.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Network error contacting the Claude API.")
    print("\nSources:")
    for i, (chunk, _) in enumerate(hits, start=1):
        print(f"  [{i}] {chunk.id}")


def main(argv: list[str] | None = None) -> None:
    from .generator import DEFAULT_MODEL

    parser = argparse.ArgumentParser(prog="tiny-rag", description="Tiny retrieval-augmented generation CLI.")
    parser.add_argument("--index", default=DEFAULT_INDEX, help=f"index file (default: {DEFAULT_INDEX})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="build an index from a file or directory")
    p.add_argument("docs")
    p.add_argument("--chunk-words", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="show the top matching chunks (no LLM call)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with Claude using retrieved context")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
