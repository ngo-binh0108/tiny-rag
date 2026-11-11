"""answer() and `tiny-rag ask` with a stand-in for the anthropic package, so no API key or network is needed."""

import sys
import types

import pytest

from tiny_rag import BM25Index, Chunk
from tiny_rag import generator
from tiny_rag.cli import main


def block(kind, text=""):
    return types.SimpleNamespace(type=kind, text=text)


def fake_anthropic(monkeypatch, stop_reason="end_turn", content=None, error=None):
    """Install a fake `anthropic` module; returns the list of requests it receives."""
    calls = []

    class APIStatusError(Exception):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.message, self.status_code = message, status_code

    module = types.ModuleType("anthropic")
    module.AuthenticationError = type("AuthenticationError", (APIStatusError,), {})
    module.RateLimitError = type("RateLimitError", (APIStatusError,), {})
    module.APIStatusError = APIStatusError
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})

    def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error(module)
        return types.SimpleNamespace(stop_reason=stop_reason, content=content or [block("text", "Answer [1].")])

    class Anthropic:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))

    module.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return calls


HITS = [(Chunk("rag.md", 0, "Retrieval grounds answers in documents."), 2.0)]


def test_answer_sends_the_numbered_context(monkeypatch):
    calls = fake_anthropic(monkeypatch)
    assert generator.answer("What grounds answers?", HITS, model="some-model", effort="low") == "Answer [1]."
    (request,) = calls
    assert request["model"] == "some-model"
    assert request["system"] == generator.SYSTEM_PROMPT
    assert "[1] (source: rag.md#0)" in request["messages"][0]["content"]


def test_answer_joins_text_blocks_only(monkeypatch):
    fake_anthropic(monkeypatch, content=[block("thinking"), block("text", " One "), block("text", "two. ")])
    assert generator.answer("q", HITS, model="m") == "One two."


def test_answer_reports_a_refusal(monkeypatch):
    fake_anthropic(monkeypatch, stop_reason="refusal", content=[])
    assert generator.answer("q", HITS, model="m") == "The model declined to answer this request."


def test_answer_marks_a_truncated_reply(monkeypatch):
    fake_anthropic(monkeypatch, stop_reason="max_tokens", content=[block("text", "Partial")])
    assert generator.answer("q", HITS, model="m").endswith("[answer truncated: max_tokens reached]")


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "i.json"
    BM25Index([HITS[0][0], Chunk("other.md", 0, "Tomatoes need sun.")]).save(path)
    return str(path)


def test_ask_prints_the_answer_and_sources(monkeypatch, capsys, index):
    fake_anthropic(monkeypatch)
    main(["--index", index, "ask", "what grounds answers", "--model", "m"])
    out = capsys.readouterr().out
    assert out.startswith("Answer [1].") and "[1] rag.md#0" in out and "other.md" not in out


def test_ask_explains_a_missing_api_key(monkeypatch, index):
    fake_anthropic(monkeypatch, error=lambda m: m.AuthenticationError("bad key", 401))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        main(["--index", index, "ask", "what grounds answers", "--model", "m"])


def test_ask_without_matches_does_not_call_the_api(monkeypatch, index):
    calls = fake_anthropic(monkeypatch)
    with pytest.raises(SystemExit, match="No relevant passages"):
        main(["--index", index, "ask", "quantum chromodynamics"])
    assert calls == []


def test_commands_need_an_index(tmp_path):
    with pytest.raises(SystemExit, match="No index at"):
        main(["--index", str(tmp_path / "missing.json"), "search", "anything"])
