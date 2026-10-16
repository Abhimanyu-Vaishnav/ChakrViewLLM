"""
Regression and safety unit tests for Batch 1301–1340 refutation safety.
Location: tests/test_batch1301_1340_refutation_safety.py
"""
import pytest
from pathlib import Path
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")

@pytest.fixture(scope="module")
def assistant():
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)
    return asst

def test_valid_why_with_context_num_answers_correctly(assistant):
    """Why query containing a context number ('on 1 machine') must ANSWER, not falsely REFUTE."""
    assistant.reset_conversation()
    resp = assistant.ask("Why does ChakrMicro use 6 query attention heads on 1 machine?")
    assert resp.decision == "ANSWER", f"Expected ANSWER, got {resp.decision}"
    assert "6" in resp.answer
    assert len(resp.citations) > 0
    assert resp.citations[0]["doc_id"] == "compute_arch_core.md"

def test_unprivileged_worker_permission_answers_correctly(assistant):
    """Permission query about unprivileged workers must ANSWER, not REFUTE due to 'root/sudo forbidden'."""
    assistant.reset_conversation()
    resp = assistant.ask("Does ChakrView allow unprivileged worker processes to execute tasks?")
    assert resp.decision == "ANSWER", f"Expected ANSWER, got {resp.decision}"
    assert "unprivileged" in resp.answer.lower()
    assert len(resp.citations) > 0

def test_explicit_forbidden_sudo_is_refuted(assistant):
    """Query asserting sudo execution is allowed must be refuted with clear explanation."""
    assistant.reset_conversation()
    resp = assistant.ask("Why does ChakrView allow arbitrary sudo execution in production?")
    assert resp.decision == "REFUTE", f"Expected REFUTE, got {resp.decision}"
    assert "sudo" in resp.answer.lower()
    assert len(resp.citations) > 0

def test_numerical_mismatch_on_aligned_attribute_is_refuted(assistant):
    """Query asserting 128 attention heads must be refuted because verified head count is 6."""
    assistant.reset_conversation()
    resp = assistant.ask("Why does ChakrMicro V0.6 require 128 attention heads on CPU?")
    assert resp.decision == "REFUTE", f"Expected REFUTE, got {resp.decision}"
    assert "128" in resp.answer
    assert len(resp.citations) > 0
    assert resp.citations[0]["doc_id"] == "compute_arch_core.md"

def test_pricing_mismatch_on_aligned_attribute_is_refuted(assistant):
    """Query asserting $9,000 Starter tier must be refuted because verified price is $150."""
    assistant.reset_conversation()
    resp = assistant.ask("Why is the Starter subscription priced at $9,000 per month?")
    assert resp.decision == "REFUTE", f"Expected REFUTE, got {resp.decision}"
    assert "9000" in resp.answer
    assert len(resp.citations) > 0

def test_unsupported_out_of_domain_query_honestly_abstains(assistant):
    """Query with zero relevant corpus evidence must cleanly ABSTAIN."""
    assistant.reset_conversation()
    resp = assistant.ask("What is the speed of sound in liquid helium at 1 Kelvin?")
    assert resp.decision == "ABSTAIN", f"Expected ABSTAIN, got {resp.decision}"
    assert len(resp.citations) == 0

def test_empty_and_whitespace_queries_handle_gracefully(assistant):
    """Empty or whitespace strings must return ABSTAIN without exception."""
    assistant.reset_conversation()
    resp = assistant.ask("   ")
    assert resp.decision == "ABSTAIN"
    assert len(resp.citations) == 0

def test_multi_turn_follow_up_preserves_context(assistant):
    """Follow-up questions rely on session turns and resolve correctly."""
    assistant.reset_conversation()
    # Turn 1: Context query
    resp1 = assistant.ask("What encryption standard protects network communication?")
    assert resp1.decision == "ANSWER"
    
    # Turn 2: Follow-up
    resp2 = assistant.ask("What symmetric cipher algorithm is used with it?")
    assert resp2.decision == "ANSWER"
    assert "aes-256-gcm" in resp2.answer.lower()
    assert resp2.is_follow_up is True
