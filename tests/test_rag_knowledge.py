"""
Unit tests for Step 11 Knowledge Ingestion, Chunking & BM25 Retrieval.
"""

from pathlib import Path
import pytest

from chakrview.runtime.knowledge import (
    KnowledgeDocument,
    KnowledgeChunk,
    KnowledgeProvenance,
    DocumentChunker,
    DocumentIngester,
    BM25KnowledgeIndex,
    LexicalRetriever,
    compute_sha256_text,
)


def test_document_chunker_deterministic():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2, min_chunk_size=3)
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16"
    chunks1 = chunker.chunk_text(text, doc_id="doc_test")
    chunks2 = chunker.chunk_text(text, doc_id="doc_test")

    assert len(chunks1) > 1
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.chunk_hash == c2.chunk_hash
        assert c1.token_count > 0


def test_document_chunker_edge_cases():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    # Empty or whitespace text
    assert chunker.chunk_text("", doc_id="doc_empty") == []
    assert chunker.chunk_text("   \n\t  ", doc_id="doc_whitespace") == []

    # Small text below chunk_size
    short_chunks = chunker.chunk_text("Short sentence here.", doc_id="doc_short")
    assert len(short_chunks) == 1
    assert short_chunks[0].chunk_id == "doc_short_c0000"
    assert short_chunks[0].text == "Short sentence here."


def test_document_ingester_text():
    ingester = DocumentIngester()
    content = "ChakrView is an indigenous neural architecture designed for sovereign intelligence."
    doc, chunks = ingester.ingest_text(
        text=content,
        title="ChakrView Overview",
        doc_id="doc_cv_01",
        source_id="src_manual",
        metadata={"category": "architecture"},
    )
    assert doc.doc_id == "doc_cv_01"
    assert doc.title == "ChakrView Overview"
    assert doc.content_hash == compute_sha256_text(content)
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc_cv_01"
    assert chunks[0].metadata["source_id"] == "src_manual"
    assert chunks[0].metadata["category"] == "architecture"


def test_document_ingester_empty_raises():
    ingester = DocumentIngester()
    with pytest.raises(ValueError):
        ingester.ingest_text("")
    with pytest.raises(ValueError):
        ingester.ingest_text("   \n ")


def test_document_ingester_files(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Plain text content about transformer decoders and RMSNorm.", encoding="utf-8")

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Markdown Notes\nRotary position embeddings RoPE theta 10000.", encoding="utf-8")

    ingester = DocumentIngester()
    doc1, chunks1 = ingester.ingest_file(txt_file)
    assert doc1.title == "sample.txt"
    assert len(chunks1) >= 1

    doc2, chunks2 = ingester.ingest_file(md_file)
    assert doc2.title == "notes.md"
    assert len(chunks2) >= 1

    # Directory ingestion
    results = ingester.ingest_directory(tmp_path)
    assert len(results) == 2


def test_document_ingester_unsupported_format(tmp_path: Path):
    bad_file = tmp_path / "data.exe"
    bad_file.write_bytes(b"\x00\x01\x02")
    ingester = DocumentIngester()
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingester.ingest_file(bad_file)


def test_bm25_knowledge_index_relevance():
    index = BM25KnowledgeIndex("test_bm25")
    chunks = [
        KnowledgeChunk("c1", "doc_A", 0, "ChakrMicro neural core causal transformer decoder with SwiGLU."),
        KnowledgeChunk("c2", "doc_A", 1, "Rotary Position Embeddings RoPE theta 10000 and Pre-RMSNorm."),
        KnowledgeChunk("c3", "doc_B", 0, "PostgreSQL relational database configuration and connections."),
    ]
    added = index.add_chunks(chunks)
    assert added == 3
    assert index.count() == 3

    # Search for transformer decoder
    results = index.search("transformer decoder", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.0

    # Search for database query
    db_results = index.search("postgresql database", top_k=2)
    assert len(db_results) > 0
    assert db_results[0][0].chunk_id == "c3"

    # Search query with no match
    empty_results = index.search("completely non-existent quantum gravity term", top_k=3)
    assert len(empty_results) == 0


def test_bm25_document_removal_and_clear():
    index = BM25KnowledgeIndex("test_bm25_removal")
    chunks = [
        KnowledgeChunk("c1", "doc_A", 0, "First chunk of document A"),
        KnowledgeChunk("c2", "doc_A", 1, "Second chunk of document A"),
        KnowledgeChunk("c3", "doc_B", 0, "Chunk of document B"),
    ]
    index.add_chunks(chunks)
    assert index.count() == 3

    removed = index.remove_document("doc_A")
    assert removed == 2
    assert index.count() == 1

    index.clear()
    assert index.count() == 0
    assert index.search("document", top_k=5) == []


def test_lexical_retriever_filtering():
    index = BM25KnowledgeIndex("test_retriever")
    index.add_chunks([
        KnowledgeChunk("c1", "doc_1", 0, "Python programming language functions, classes, and decorators."),
        KnowledgeChunk("c2", "doc_2", 0, "Rust memory safety borrow checker and lifetimes."),
    ])
    retriever = LexicalRetriever(index, min_score=0.1)
    results = retriever.retrieve("Python functions", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"


def test_knowledge_provenance():
    prov = KnowledgeProvenance(
        doc_id="doc_01",
        chunk_id="chunk_01",
        source_id="src_manual",
        content_hash="abc123hash",
        retrieval_score=1.45,
        text_preview="Snippet preview",
        metadata={"file": "guide.md"},
    )
    d = prov.to_dict()
    assert d["doc_id"] == "doc_01"
    assert d["retrieval_score"] == 1.45
    restored = KnowledgeProvenance.from_dict(d)
    assert restored.chunk_id == "chunk_01"
    assert restored.metadata["file"] == "guide.md"
