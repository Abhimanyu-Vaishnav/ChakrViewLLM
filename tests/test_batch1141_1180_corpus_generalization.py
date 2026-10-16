"""
Unit and Integration Tests for Batches 1141-1180.
Location: tests/test_batch1141_1180_corpus_generalization.py

Verifies:
1. Multi-document 52-corpus ingestion and metadata tracking.
2. Embedding LRU caching and bit-for-bit equivalence.
3. Multi-turn follow-up resolution without false pronoun triggers.
4. Distractor discrimination across identical numeric values.
5. Ingest invalidation on file modification and deletion.
6. Out-of-domain honest abstention.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant
from chakrview.runtime.retrieval import CachedEmbeddingProvider, DeterministicHashEmbeddingProvider


CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")


def test_large_corpus_ingestion():
    """Verifies that the 52-document corpus ingests and tracks metadata accurately."""
    ast = LocalKnowledgeAssistant(chunk_size=800, chunk_overlap=100)
    count = ast.ingest_directory(CORPUS_DIR)
    assert count >= 52
    assert len(ast.doc_metadata) == 52
    assert "compute_arch_core.md" in ast.doc_metadata
    assert "billing_subscription_tiers.md" in ast.doc_metadata


def test_cached_embedding_bit_exactness():
    """Verifies that CachedEmbeddingProvider matches uncached execution bit-for-bit."""
    base_prov = DeterministicHashEmbeddingProvider()
    cached_prov = CachedEmbeddingProvider(base_prov, max_capacity=100)

    sample_texts = [
        "What is the hidden dimension of ChakrMicro V0.6?",
        "How many query heads are configured in grouped-query attention?",
        "What encryption cipher protects in-transit wire communication?",
    ]

    for t in sample_texts:
        v_uncached = base_prov.embed_text(t)
        v_cached = cached_prov.embed_text(t)
        v_repeat = cached_prov.embed_text(t)
        assert v_uncached == v_cached
        assert v_cached == v_repeat

    assert cached_prov.hits == 3
    assert cached_prov.misses == 3
    assert cached_prov.size == 3


def test_followup_resolution_and_negative_pronoun_filter():
    """Verifies that follow-up resolves correctly and doesn't falsely trigger on substrings."""
    ast = LocalKnowledgeAssistant(extract_concise_span=True)
    ast.ingest_directory(CORPUS_DIR)

    ast.reset_conversation()
    r1 = ast.ask("What is the monthly fee for the Starter subscription tier?")
    assert r1.decision == "ANSWER"
    assert "150" in r1.answer

    # True follow-up with pronoun 'that tier'
    r2 = ast.ask("How many concurrent worker processes are included in that tier?")
    assert r2.is_follow_up is True
    assert r2.decision == "ANSWER"
    assert "2" in r2.answer

    # Negative test: Unrelated question containing 'it' inside a word ('capital')
    r3 = ast.ask("What is the capital of France?")
    assert r3.is_follow_up is False
    assert r3.decision == "ABSTAIN"


def test_distractor_number_discrimination():
    """Verifies that identical numbers in different contexts are discriminated correctly."""
    ast = LocalKnowledgeAssistant(extract_concise_span=True)
    ast.ingest_directory(CORPUS_DIR)

    # 512 in architecture vs 512 in quantization calibration
    r_arch = ast.ask("What is the primary context ceiling in tokens for ChakrMicro V0.6?")
    assert r_arch.decision == "ANSWER"
    assert "512" in r_arch.answer
    assert r_arch.citations[0]["doc_id"] == "compute_arch_core.md"

    r_quant = ast.ask("How many domain tokens are in the calibration dataset for quantization?")
    assert r_quant.decision == "ANSWER"
    assert "512" in r_quant.answer
    assert r_quant.citations[0]["doc_id"] == "compute_quantization.md"


def test_document_modification_reingest(tmp_path: Path):
    """Verifies that document updates invalidate cache and re-index properly."""
    doc_file = tmp_path / "dynamic_doc.md"
    doc_file.write_text("# Dynamic Version\n- Version Code: V1_INITIAL\n", encoding="utf-8")

    ast = LocalKnowledgeAssistant(extract_concise_span=True, use_embedding_cache=True)
    ast.ingest_document(doc_file)

    r1 = ast.ask("What is the version code of the dynamic document?")
    assert "V1_INITIAL" in r1.answer

    # Update file content
    doc_file.write_text("# Dynamic Version\n- Version Code: V2_UPDATED\n", encoding="utf-8")
    ast.ingest_document(doc_file)

    r2 = ast.ask("What is the version code of the dynamic document?")
    assert "V2_UPDATED" in r2.answer
    assert "V1_INITIAL" not in r2.answer


def test_unindexed_topic_abstention():
    """Verifies that out-of-domain unindexed questions reliably trigger abstention."""
    ast = LocalKnowledgeAssistant(extract_concise_span=True)
    ast.ingest_directory(CORPUS_DIR)

    resp = ast.ask("What is the orbital eccentricity of Jupiter moon Europa?")
    assert resp.decision == "ABSTAIN"
    assert "could not find sufficient" in resp.answer.lower()
    assert len(resp.citations) == 0
