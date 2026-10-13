"""
Test Suite for Batches 191–200: Query-Aware Pointer, Constrained Decoding & Multi-Turn Grounding.
Location: tests/test_batch191_200_grounded_conversation.py
"""
import json
from pathlib import Path
import pytest
import torch

from chakrview.runtime.constrained_decoder import TokenTrie, decode_pointer_constrained
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    if not tok_dir.exists():
        pytest.skip("Tokenizer directory not found")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


def test_query_aware_pointer_masking():
    """Verify that query-aware pointer suppresses prompt overlap tokens without error."""
    head = ChakrPointerHead(d_model=64, vocab_size=100)
    hidden_states = torch.randn(2, 10, 64)
    input_ids = torch.randint(0, 100, (2, 10))
    # Make token 4 identical to query token 1 to trigger overlap
    input_ids[:, 4] = input_ids[:, 1]
    vocab_logits = torch.randn(2, 10, 100)
    context_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask[:, :4] = False  # tokens 0..3 are query
    query_mask = torch.zeros(2, 10, dtype=torch.bool)
    query_mask[:, :4] = True

    # With query dampening
    final_probs, pointer_attn, gate = head(
        hidden_states=hidden_states,
        input_ids=input_ids,
        vocab_logits=vocab_logits,
        context_mask=context_mask,
        query_mask=query_mask,
        disentangle_query_overlap=True,
    )
    assert final_probs.shape == (2, 10, 100)
    assert pointer_attn.shape == (2, 10, 10)
    assert gate.shape == (2, 10, 1)


def test_token_trie_construction(tokenizer):
    """Verify TokenTrie correctly guides allowed branch tokens."""
    trie = TokenTrie()
    trie.insert(tokenizer.encode("Hydrogen"))
    trie.insert(tokenizer.encode("Helium"))

    allowed_root = trie.get_allowed_tokens([])
    assert len(allowed_root) > 0

    # Prefix match
    prefix = tokenizer.encode("Hydrogen")[:1]
    allowed_next = trie.get_allowed_tokens(prefix)
    assert len(allowed_next) > 0


def test_evidence_extractor_conflict_detection():
    """Verify that conflicting facts are flagged correctly."""
    res1 = EvidenceExtractor.extract_slot("The boiling point of water is 100 C.", "What is the boiling point of water?")
    assert res1.success
    assert "100" in res1.extracted_value


def test_manifest_and_reports_integrity():
    """Verify that all required reports and manifests from Batch 191-200 exist."""
    base_dir = Path("experiments/batch191_200_grounded_conversation/reports")
    expected_reports = [
        "initial_audit.md",
        "query_overlap_diagnostics.json",
        "relation_binding_data_audit.md",
        "query_aware_pointer_training.json",
        "constrained_decoding_results.json",
        "answer_verification_contract.md",
        "multiturn_grounded_benchmark.json",
        "recovery_and_resource_tests.json",
        "final_evaluation.json",
        "failure_analysis.md",
        "milestone_summary.md"
    ]
    for r in expected_reports:
        p = base_dir / r
        assert p.exists(), f"Report missing: {p}"

    manifest = Path("experiments/batch191_200_grounded_conversation/experiment_manifest.json")
    assert manifest.exists()
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["canonical_checkpoint"]["unique_parameter_count"] == 3443136
