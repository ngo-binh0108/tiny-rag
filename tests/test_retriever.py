from tiny_rag import BM25Index, Chunk, chunk_text, tokenize
from tiny_rag.cli import main
from tiny_rag.generator import build_prompt


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Transformer, really?") == ["transformer", "really"]


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, "doc.txt", max_words=100, overlap=20)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_bm25_ranks_relevant_chunk_first():
    idx = BM25Index([
        Chunk("a", 0, "Cats are small domesticated carnivorous mammals."),
        Chunk("b", 0, "Attention lets transformers weigh tokens in a sequence."),
        Chunk("c", 0, "Embeddings map tokens to dense vectors."),
    ])
    hits = idx.search("how does attention work in transformers", k=2)
    assert hits[0][0].source == "b"
    assert idx.search("quantum chromodynamics") == []


def test_save_load_roundtrip(tmp_path):
    idx = BM25Index([Chunk("x", 0, "retrieval augmented generation")])
    path = tmp_path / "idx.json"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert loaded.chunks == idx.chunks
    assert loaded.search("retrieval")[0][0].id == "x#0"


def test_build_prompt_numbers_passages():
    prompt = build_prompt("q?", [(Chunk("s", 2, "hello"), 1.0)])
    assert "[1] (source: s#2)" in prompt and "Question: q?" in prompt


def test_cli_index_and_search(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("Retrieval augmented generation grounds answers in documents.")
    (docs / "other.md").write_text("Gardening tips for tomatoes.")
    index = str(tmp_path / "i.json")
    main(["--index", index, "index", str(docs)])
    main(["--index", index, "search", "retrieval grounding"])
    out = capsys.readouterr().out
    assert "Indexed 2 chunks" in out
    assert "1. rag.md#0" in out


def test_chunk_overlap_must_be_smaller_than_the_chunk():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", "doc.txt", max_words=10, overlap=10)


def test_short_paragraphs_share_a_chunk():
    chunks = chunk_text("one two\n\nthree four\n\n\nfive", "doc.md", max_words=10, overlap=2)
    assert [c.text for c in chunks] == ["one two three four five"]


def test_load_documents_reads_only_text_files(tmp_path):
    from tiny_rag.retriever import load_documents

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.md").write_text("hello", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert sorted(load_documents(tmp_path)) == ["data.csv", "sub/notes.md"]
    assert load_documents(tmp_path / "sub" / "notes.md") == {"notes.md": "hello"}


def test_index_of_an_empty_folder_stops(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="No text documents"):
        main(["--index", str(tmp_path / "i.json"), "index", str(tmp_path)])
