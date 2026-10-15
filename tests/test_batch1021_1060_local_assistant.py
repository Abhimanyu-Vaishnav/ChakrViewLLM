"""
Unit and integration tests for LocalKnowledgeAssistant MVP.
"""
import pytest
import tempfile
import os
from pathlib import Path

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant


@pytest.fixture
def sample_assistant():
    with tempfile.TemporaryDirectory() as tmpdir:
        doc1 = Path(tmpdir) / "system_spec.md"
        doc1.write_text(
            "# Architecture Specification\n\n"
            "ChakrMicro uses an autoregressive causal transformer with 6 layers and 6 heads.\n"
            "The maximum context ceiling is 512 tokens.\n\n"
            "All tool executions run within an isolated sandboxed subprocess.",
            encoding="utf-8"
        )

        doc2 = Path(tmpdir) / "security_policy.txt"
        doc2.write_text(
            "Access Policy:\n"
            "Production database requires MFA and OAuth2 tokens.\n"
            "Arbitrary shell commands are permanently disabled for safety.",
            encoding="utf-8"
        )

        assistant = LocalKnowledgeAssistant()
        assistant.ingest_directory(tmpdir)
        yield assistant


def test_document_ingestion_and_metadata(sample_assistant):
    docs = sample_assistant.list_indexed_documents()
    assert len(docs) == 2
    doc_ids = {d["doc_id"] for d in docs}
    assert "system_spec.md" in doc_ids
    assert "security_policy.txt" in doc_ids


def test_grounded_answer_with_citations(sample_assistant):
    resp = sample_assistant.ask("What is the context ceiling of ChakrMicro?")
    assert resp.decision == "ANSWER"
    assert "512 tokens" in resp.answer
    assert "[Source: system_spec.md" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0]["doc_id"] == "system_spec.md"


def test_honest_abstention_on_unindexed_topic(sample_assistant):
    resp = sample_assistant.ask("What is the GDP of Germany in 2024?")
    assert resp.decision == "ABSTAIN"
    assert "could not find sufficient verified evidence" in resp.answer
    assert len(resp.citations) == 0


def test_multi_turn_followup(sample_assistant):
    # First turn
    resp1 = sample_assistant.ask("What is the context ceiling of ChakrMicro?")
    assert resp1.decision == "ANSWER"

    # Follow-up referring to the document
    resp2 = sample_assistant.ask("How many layers does that model have?")
    assert resp2.is_follow_up is True
    assert resp2.decision == "ANSWER"
    assert "6 layers" in resp2.answer
    assert "system_spec.md" in resp2.answer


def test_deleted_document_pruning_on_reindex():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        doc1 = d / "doc1.md"
        doc1.write_text("# Doc 1\nAlpha value is 100.\n", encoding="utf-8")
        doc2 = d / "doc2.md"
        doc2.write_text("# Doc 2\nBeta value is 200.\n", encoding="utf-8")

        assistant = LocalKnowledgeAssistant()
        assistant.ingest_directory(d)
        assert len(assistant.list_indexed_documents()) == 2

        # Delete doc2 from filesystem and re-ingest directory
        os.remove(doc2)
        assistant.ingest_directory(d)

        docs = assistant.list_indexed_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc1.md"

        # Querying deleted content now abstains
        resp = assistant.ask("What is Beta quantity for doc2?")
        assert resp.decision == "ABSTAIN"

