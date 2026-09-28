"""
Unit tests for Step 9 Knowledge Layer abstractions (chakrview.runtime.knowledge).
"""

import pytest
from chakrview.runtime.knowledge import (
    KnowledgeSource,
    KnowledgeDocument,
    KnowledgeChunk,
    InMemoryKnowledgeIndex,
    SimpleRetriever,
    ContextProvider,
    compute_sha256_text,
)


def test_knowledge_source_lifecycle():
    source = KnowledgeSource(
        source_id="src_manual_v1",
        source_type="file",
        uri="docs/manual.pdf",
        title="Technical Manual",
        metadata={"department": "engineering"},
    )
    d = source.to_dict()
    assert d["source_id"] == "src_manual_v1"
    assert d["source_type"] == "file"
    
    restored = KnowledgeSource.from_dict(d)
    assert restored.title == "Technical Manual"
    assert restored.metadata["department"] == "engineering"


def test_knowledge_document_hash():
    doc1 = KnowledgeDocument(
        doc_id="doc_001",
        source_id="src_manual_v1",
        title="Section 1",
        content="This is the operating manual for ChakrView architecture.",
    )
    assert len(doc1.content_hash) == 64
    assert doc1.content_hash == compute_sha256_text(doc1.content)

    doc2 = KnowledgeDocument.from_dict(doc1.to_dict())
    assert doc2.content_hash == doc1.content_hash


def test_knowledge_chunk_creation():
    chunk = KnowledgeChunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        chunk_index=0,
        text="ChakrView uses a decoupled cognitive brain core.",
        token_count=8,
    )
    assert len(chunk.chunk_hash) == 64
    assert chunk.token_count == 8
    
    chunk_dict = chunk.to_dict()
    restored = KnowledgeChunk.from_dict(chunk_dict)
    assert restored.chunk_id == "chunk_001"
    assert restored.chunk_hash == chunk.chunk_hash


def test_in_memory_knowledge_index():
    index = InMemoryKnowledgeIndex("test_index")
    chunks = [
        KnowledgeChunk("c1", "doc_A", 0, "ChakrMicro neural core causal transformer decoder"),
        KnowledgeChunk("c2", "doc_A", 1, "Pre-RMSNorm and Rotary Position Embeddings RoPE"),
        KnowledgeChunk("c3", "doc_B", 0, "PostgreSQL database connection parameters and ports"),
    ]
    added = index.add_chunks(chunks)
    assert added == 3
    assert index.count() == 3

    # Search query matching c1
    results = index.search("transformer decoder", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.0

    # Test removing document doc_A
    removed = index.remove_document("doc_A")
    assert removed == 2
    assert index.count() == 1
    assert index.search("transformer", top_k=2) == []

    # Clear
    index.clear()
    assert index.count() == 0


def test_context_provider_budget():
    index = InMemoryKnowledgeIndex()
    index.add_chunks([
        KnowledgeChunk("c1", "doc_A", 0, "Alpha beta gamma delta epsilon", token_count=10),
        KnowledgeChunk("c2", "doc_A", 1, "Zeta eta theta iota kappa", token_count=10),
        KnowledgeChunk("c3", "doc_A", 2, "Lambda mu nu xi omicron", token_count=10),
    ])
    retriever = SimpleRetriever(index)
    
    # Provider with tight budget of 15 tokens -> only 1 chunk fits
    provider = ContextProvider(retriever, max_context_tokens=15)
    context, included = provider.build_context("Alpha Zeta Lambda", top_k=3)
    
    assert len(included) == 1
    assert included[0].chunk_id in ("c1", "c2", "c3")
    assert "--- KNOWLEDGE CONTEXT ---" in context
    assert "-------------------------" in context
