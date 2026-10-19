"""
Unit and Integration Tests for Ephemeral Conversation Memory & Multi-Turn Context.
Location: tests/test_conversation_memory.py

Validates all 10 Phase 3 mission requirements:
1. Store the fact "The project codename is Blue Mango 47."
2. Retrieve that fact in a subsequent turn.
3. Preserve context across an unrelated calculator turn.
4. Correctly answer "What was the project codename I told you earlier?"
5. Keep a fresh session isolated from the previous session.
6. Continue abstaining on unknown factual questions without evidence.
7. Preserve document citations for grounded QA.
8. Ensure persistent memory is not written when consent is inactive.
9. Ensure calculator routing still works.
10. Ensure the neural backend is not falsely attributed if it was not invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from chakrview.runtime.conversation_memory import EphemeralConversationMemory
from experiments.batch2221_2260.integrated_assistant_adapter import IntegratedAssistantAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_01_store_fact_statement():
    """Requirement 1: Store the fact 'The project codename is Blue Mango 47.'"""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask("The project codename is Blue Mango 47.")

    assert resp.decision == "STATEMENT"
    assert "project codename" in resp.answer.lower()
    assert "Blue Mango 47" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "working_memory"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"
    assert "project codename" in adapter.working_memory.facts


def test_02_retrieve_fact_subsequent_turn():
    """Requirement 2: Retrieve that fact in a subsequent turn."""
    adapter = IntegratedAssistantAdapter()
    adapter.ask("The project codename is Blue Mango 47.")
    
    resp = adapter.ask("What is the project codename?")
    assert resp.decision == "WORKING_MEMORY"
    assert "Blue Mango 47" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "working_memory"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_03_preserve_context_across_unrelated_calculator_turn():
    """Requirement 3: Preserve context across an unrelated calculator turn."""
    adapter = IntegratedAssistantAdapter()
    # Turn 1: Fact
    adapter.ask("The project codename is Blue Mango 47.")
    # Turn 2: Unrelated calculator
    resp_calc = adapter.ask("What is 2 + 2?")
    assert resp_calc.decision == "CALCULATED"
    assert "4" in resp_calc.answer
    assert resp_calc.execution_breakdown["subsystem"] == "safe_calculator"

    # Context verification: Fact is still retained
    assert "project codename" in adapter.working_memory.facts
    assert adapter.working_memory.facts["project codename"].value == "Blue Mango 47"


def test_04_answer_exact_conversational_sequence():
    """Requirement 4: Correctly answer 'What was the project codename I told you earlier?' in 3-turn sequence."""
    adapter = IntegratedAssistantAdapter()
    
    # 1. User: "The project codename is Blue Mango 47."
    r1 = adapter.ask("The project codename is Blue Mango 47.")
    assert r1.decision == "STATEMENT"
    assert "Blue Mango 47" in r1.answer

    # 2. User: "What is 2 + 2?"
    r2 = adapter.ask("What is 2 + 2?")
    assert r2.decision == "CALCULATED"
    assert "4" in r2.answer

    # 3. User: "What was the project codename I told you earlier?"
    r3 = adapter.ask("What was the project codename I told you earlier?")
    assert r3.decision == "WORKING_MEMORY"
    assert "Blue Mango 47" in r3.answer
    assert r3.execution_breakdown["subsystem"] == "working_memory"
    assert r3.execution_breakdown["neural_inference"] == "not invoked"


def test_05_session_isolation():
    """Requirement 5: Keep a fresh session isolated from the previous session."""
    adapter = IntegratedAssistantAdapter()
    adapter.ask("The project codename is Blue Mango 47.")
    assert "project codename" in adapter.working_memory.facts

    # Fresh session / reset
    adapter.reset_session()
    assert len(adapter.working_memory.facts) == 0
    assert len(adapter.working_memory.turns) == 0

    # Querying the reset session must not recall the fact and must abstain
    r_after = adapter.ask("What was the project codename I told you earlier?")
    assert r_after.decision == "ABSTAIN"
    assert "could not find sufficient verified evidence" in r_after.answer


def test_06_abstain_on_unknown_factual_questions():
    """Requirement 6: Continue abstaining on unknown factual questions without evidence."""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask("What is the secret launch date of Project Apollo 99?")
    assert resp.decision == "ABSTAIN"
    assert "could not find sufficient verified evidence" in resp.answer
    assert len(resp.citations) == 0


def test_07_preserve_document_citations_for_grounded_qa():
    """Requirement 7: Preserve document citations for grounded QA."""
    adapter = IntegratedAssistantAdapter()
    doc_dir = REPO_ROOT / "experiments" / "batch1021_1060" / "data" / "sample_corpus"
    assert doc_dir.exists()
    adapter.ingest_directory(doc_dir)

    resp = adapter.ask("What is the availability target in billing terms SLA?")
    assert resp.decision == "ANSWER"
    assert len(resp.citations) > 0
    assert resp.citations[0]["doc_id"] == "billing_terms.md"
    assert "99.95%" in resp.answer


def test_08_persistent_memory_not_written_without_consent(tmp_path):
    """Requirement 8: Ensure persistent memory is not written when consent is inactive."""
    mem_file = tmp_path / "ephemeral_test_user_memory.json"
    fb_file = tmp_path / "ephemeral_test_user_feedback.json"

    adapter_no_consent = IntegratedAssistantAdapter(
        memory_path=mem_file,
        feedback_path=fb_file,
        enable_consent_default=False
    )
    # Storing fact in ephemeral memory
    resp = adapter_no_consent.ask("The project codename is Blue Mango 47.")
    assert resp.decision == "STATEMENT"
    assert resp.memory_logged is False

    # Verify no file written to disk
    assert not mem_file.exists()


def test_09_calculator_routing_still_works():
    """Requirement 9: Ensure calculator routing still works."""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask("What is 500 - 234?")
    assert resp.decision == "CALCULATED"
    assert "266" in resp.answer
    assert resp.tool_trace["tool_name"] == "safe_calculator"


def test_10_neural_backend_not_falsely_attributed():
    """Requirement 10: Ensure the neural backend is not falsely attributed if it was not invoked."""
    adapter = IntegratedAssistantAdapter()
    
    # Statement
    r1 = adapter.ask("The server IP is 192.168.1.100")
    assert r1.execution_breakdown["neural_inference"] == "not invoked"

    # Calculation
    r2 = adapter.ask("What is 100 / 4?")
    assert r2.execution_breakdown["neural_inference"] == "not invoked"

    # Memory recall
    r3 = adapter.ask("What is the server IP?")
    assert r3.execution_breakdown["neural_inference"] == "not invoked"

    # Conversational greeting
    r4 = adapter.ask("Hello there!")
    assert r4.decision == "CONVERSATIONAL"
    assert r4.execution_breakdown["neural_inference"] == "not invoked"
