"""Document loading, chunking, and BM25 retrieval — pure Python, no dependencies."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".rst", ".py", ".json", ".csv", ".html"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with what which who how why when where do does".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, and drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class Chunk:
    source: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def chunk_text(text: str, source: str, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split text into overlapping word windows, preferring paragraph boundaries."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    paragraphs = [p.split() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    window: list[str] = []
    for words in paragraphs:
        if window and len(window) + len(words) > max_words:
            chunks.append(Chunk(source, len(chunks), " ".join(window)))
            window = window[-overlap:] if overlap else []
        window.extend(words)
        while len(window) > max_words:
            chunks.append(Chunk(source, len(chunks), " ".join(window[:max_words])))
            window = window[max_words - overlap :]
    if window:
        chunks.append(Chunk(source, len(chunks), " ".join(window)))
    return chunks


def load_documents(path: str | Path) -> dict[str, str]:
    """Read a file, or every text-like file under a directory."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
    docs = {}
    for f in files:
        name = f.name if root.is_file() else f.relative_to(root).as_posix()
        docs[name] = f.read_text(encoding="utf-8", errors="replace")
    return docs


@dataclass
class BM25Index:
    """Okapi BM25 over a list of chunks."""

    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75
    _tf: list[Counter] = field(init=False, repr=False)
    _df: Counter = field(init=False, repr=False)
    _lengths: list[int] = field(init=False, repr=False)
    _avgdl: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self.chunks), self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, dl in zip(self.chunks, self._tf, self._lengths):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                    score += self._idf(t) * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @classmethod
    def from_path(cls, path: str | Path, max_words: int = 200, overlap: int = 40) -> "BM25Index":
        chunks: list[Chunk] = []
        for name, text in load_documents(path).items():
            chunks.extend(chunk_text(text, name, max_words, overlap))
        return cls(chunks)

    def save(self, path: str | Path) -> None:
        data = {"k1": self.k1, "b": self.b, "chunks": [asdict(c) for c in self.chunks]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in data["chunks"]], k1=data["k1"], b=data["b"])
