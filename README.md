# tiny-rag

A minimal **retrieval-augmented generation (RAG)** command-line tool for Python.

- **Retrieval:** pure-Python BM25 with paragraph-aware chunking. No dependencies, no embedding model, no vector database.
- **Generation:** answers from [Claude](https://www.anthropic.com/claude) with numbered citations back to the source chunks.
- **Small:** about 300 lines of readable code, a reasonable starting point for learning how RAG works.

## Install

```bash
git clone https://github.com/ngo-binh0108/tiny-rag.git
cd tiny-rag
pip install -e ".[llm]"      # add ",dev" for pytest
```

## Usage

```bash
# 1. Build an index from a file or a folder of .md/.txt/.py/... files
tiny-rag index examples/docs

# 2. Inspect retrieval (no API call)
tiny-rag search "how does self-attention work"

# 3. Ask a question; Claude answers using only the retrieved passages
export ANTHROPIC_API_KEY=sk-ant-...
tiny-rag ask "Why does RAG reduce hallucination?"
```

Example output:

```
RAG reduces hallucination because the model grounds its answer in retrieved
passages rather than relying only on what it memorized during training [1].

Sources:
  [1] rag.md#0
  [2] transformers.md#0
```

### Options

| Flag | Command | Default | Description |
|---|---|---|---|
| `--index PATH` | all | `.tiny_rag_index.json` | Where the index is stored |
| `--chunk-words N` | `index` | `200` | Max words per chunk |
| `--overlap N` | `index` | `40` | Words shared between consecutive chunks |
| `-k N` | `search`, `ask` | `4` | Number of chunks to retrieve |
| `--model ID` | `ask` | `claude-opus-5-5` | Claude model (or set `TINY_RAG_MODEL`) |
| `--effort LEVEL` | `ask` | `medium` | `low`, `medium`, `high`, `xhigh`, `max` |

## Use as a library

```python
from tiny_rag import BM25Index
from tiny_rag.generator import answer

index = BM25Index.from_path("examples/docs")
hits = index.search("what is multi-head attention", k=3)
print(answer("What is multi-head attention?", hits))
```

## How it works

1. **Chunking.** Documents are split into overlapping word windows that break at paragraph boundaries where possible.
2. **Indexing.** Each chunk's term frequencies are stored. BM25 scores combine term frequency, inverse document frequency, and length normalization.
3. **Retrieval.** The top-k chunks for the question are selected.
4. **Generation.** The chunks are numbered and sent to Claude with instructions to answer only from them, cite the passages it uses, and say so when the answer isn't there.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
