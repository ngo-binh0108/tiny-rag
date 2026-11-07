"""Answer generation with Claude, grounded in retrieved chunks."""

from __future__ import annotations

import os

from .retriever import Chunk

DEFAULT_MODEL = os.environ.get("TINY_RAG_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you rely on with bracketed numbers like [1] or [2][3]. "
    "If the context does not contain the answer, say so plainly instead of guessing."
)


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    passages = "\n\n".join(
        f"[{i}] (source: {chunk.id})\n{chunk.text}" for i, (chunk, _) in enumerate(hits, start=1)
    )
    return f"<context>\n{passages}\n</context>\n\nQuestion: {question}"


def answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
) -> str:
    """Ask Claude to answer `question` from `hits`. Requires the `anthropic` package."""
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {}
    if model.startswith(("claude-opus-5", "claude-fable-5-1")):
        # Server-side refusal fallback: re-run on a fallback model if the request is declined.
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n[answer truncated: max_tokens reached]"
    return text
