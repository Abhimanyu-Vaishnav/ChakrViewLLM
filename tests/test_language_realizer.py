"""
Focused Tests for ChakrView Language Understanding, Intent & Natural Realization.

Verifies:
1. Language classification: English, Hindi (Devanagari), Hinglish (Romanized Hindi).
2. Intent classification: Greeting, Social check-in, Proposition, Factual query, Task, Clarification.
3. Natural Language Realization:
   - Conversational greeting and social check-in
   - Proposition confirmation ("Got it. Ram lives in Delhi.")
   - Memory-based factual answering ("Ram lives in Delhi.")
   - Honest abstention for unknown entities
   - Conflict reporting for contradictory evidence
4. Conversational context reference resolution ("Where does he live?", "What about his city?").
5. Canonical baseline immutability (3,443,136 parameters, Delta W = 0).
"""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import (
    verify_canonical_baseline,
    CANONICAL_BASELINE_SHA256,
    CANONICAL_PARAM_COUNT,
)
from chakrview.cognition.language_realizer import (
    LanguageClassifier,
    IntentClassifier,
    LanguageRealizer,
    LanguageTag,
    ConversationalIntent,
    ConversationalContextTracker,
    ConversationalTurn,
)
from scripts.founder_manual_test import ChakrViewRuntimeSession


def test_language_classification_multilingual():
    """Verify distinct classification of English, Hindi, and Hinglish."""
    # English
    assert LanguageClassifier.classify("Ram lives in Delhi.") == LanguageTag.ENGLISH
    assert LanguageClassifier.classify("How are you today?") == LanguageTag.ENGLISH

    # Hindi (Devanagari)
    assert LanguageClassifier.classify("राम दिल्ली में रहता है।") == LanguageTag.HINDI
    assert LanguageClassifier.classify("आप कैसे हैं?") == LanguageTag.HINDI

    # Hinglish (Romanized Hindi)
    assert LanguageClassifier.classify("Tum kaise ho") == LanguageTag.HINGLISH
    assert LanguageClassifier.classify("Ram Delhi mein rehta hai") == LanguageTag.HINGLISH
    assert LanguageClassifier.classify("Aap kahan rehte hain?") == LanguageTag.HINGLISH


def test_intent_classification():
    """Verify conversational, factual, task, and proposition intent discrimination."""
    # Social check-in
    assert IntentClassifier.classify("How are you", LanguageTag.ENGLISH) == ConversationalIntent.SOCIAL_CHECK_IN
    assert IntentClassifier.classify("Tum kaise ho", LanguageTag.HINGLISH) == ConversationalIntent.SOCIAL_CHECK_IN

    # Greeting
    assert IntentClassifier.classify("Hello there", LanguageTag.ENGLISH) == ConversationalIntent.GREETING
    assert IntentClassifier.classify("Namaste", LanguageTag.HINGLISH) == ConversationalIntent.GREETING

    # Task request
    assert IntentClassifier.classify("Create a step-by-step plan to move to Delhi", LanguageTag.ENGLISH) == ConversationalIntent.TASK_REQUEST

    # Proposition
    assert IntentClassifier.classify("Ram lives in Delhi.", LanguageTag.ENGLISH) == ConversationalIntent.PROPOSITION

    # Incomplete statement / clarification
    assert IntentClassifier.classify("Computer is", LanguageTag.ENGLISH) == ConversationalIntent.CLARIFICATION


def test_natural_language_realization_social_and_greeting():
    """Verify natural conversational responses without technical diagnostic exposure."""
    session = ChakrViewRuntimeSession()

    res_how_are_you = session.process_input("How are you")
    assert "functioning well" in res_how_are_you["final_response"]
    assert "DECISION" not in res_how_are_you["final_response"]

    res_tum_kaise_ho = session.process_input("Tum kaise ho")
    assert "Main theek hoon" in res_tum_kaise_ho["final_response"]
    assert res_tum_kaise_ho["detected_language"] == "HINGLISH"


def test_natural_language_realization_proposition_and_recall():
    """Verify natural confirmation of propositions and natural query answering."""
    session = ChakrViewRuntimeSession()

    # Turn 1: Store proposition
    t1 = session.process_input("Ram lives in Delhi.")
    assert "Got it. Ram lives in Delhi." in t1["final_response"]

    # Turn 2: Query memory
    t2 = session.process_input("Where does Ram live?")
    assert "Ram lives in Delhi." in t2["final_response"]
    assert "--[LIVES_IN]-->" not in t2["final_response"]


def test_conversational_context_reference_resolution():
    """Verify multi-turn reference resolution for pronouns and contextual references."""
    session = ChakrViewRuntimeSession()

    # Step 1: Establish subject
    session.process_input("Ram lives in Delhi.")

    # Step 2: Use pronoun "he"
    t_pronoun = session.process_input("Where does he live?")
    assert "Ram lives in Delhi." in t_pronoun["final_response"]

    # Step 3: Contextual reference "his city"
    t_city = session.process_input("What about his city?")
    assert "Ram lives in Delhi." in t_city["final_response"]


def test_truthful_realization_of_abstention_and_contradiction():
    """Verify that honest abstention and conflicts are realized truthfully."""
    session = ChakrViewRuntimeSession()

    # Unknown entity -> Honest abstention
    res_unknown = session.process_input("Who is Zorvax-917 and where does it live?")
    assert "I do not have verified evidence" in res_unknown["final_response"]
    assert res_unknown["confidence"] == 0.0

    # Contradiction intervention
    session.process_input("Ram lives in Delhi.")
    res_conflict = session.process_input("Ram lives in Delhi.", active_evidence={"ram::lives_in": "Mumbai"})
    assert "conflict" in res_conflict["final_response"].lower() or "virodhabhas" in res_conflict["final_response"].lower()
    assert res_conflict["decision"] == "REPORT_CONFLICT"


def test_definition_query_and_incomplete_routing():
    """
    Verify all 6 core test prompts from founder review:
    1. 'Computer is' -> CLARIFICATION / INCOMPLETE
    2. 'Define computer' -> DEFINITION_QUERY
    3. 'Tell me the definition of a computer' -> DEFINITION_QUERY
    4. 'Mujhe computer ki definition batao' -> DEFINITION_QUERY
    5. 'Main bhi theek hoon, zara mujhe computer ki definition batao' -> HINGLISH / DEFINITION_QUERY
    6. 'Main bhi thik hu, jra muje computer ki defination btao' -> HINGLISH / DEFINITION_QUERY
    7. 'How are you?' -> SOCIAL_CHECK_IN
    8. 'Ram lives in Delhi.' -> PROPOSITION
    """
    session = ChakrViewRuntimeSession()

    # 1. Incomplete statement
    r_inc = session.process_input("Computer is")
    assert r_inc["intent"] == ConversationalIntent.CLARIFICATION
    assert "incomplete" in r_inc["final_response"].lower() or "adhura" in r_inc["final_response"].lower()

    # 2. English definition queries
    for eng_q in ["Define computer", "Tell me the definition of a computer", "Tell Me Defination of Computer"]:
        r_def = session.process_input(eng_q)
        assert r_def["detected_language"] == "ENGLISH"
        assert r_def["intent"] == ConversationalIntent.DEFINITION_QUERY
        assert r_def["decision"] == "ANSWER"
        assert "electronic device" in r_def["final_response"]

    # 3. Hinglish definition queries
    for hing_q in [
        "Mujhe computer ki definition batao",
        "Main bhi theek hoon, zara mujhe computer ki definition batao",
        "Main bhi thik hu, jra muje computer ki defination btao",
    ]:
        r_hing = session.process_input(hing_q)
        assert r_hing["detected_language"] == "HINGLISH"
        assert r_hing["intent"] == ConversationalIntent.DEFINITION_QUERY
        assert r_hing["decision"] == "ANSWER"
        assert "electronic device" in r_hing["final_response"]

    # 4. Social check-in
    r_soc = session.process_input("How are you?")
    assert r_soc["intent"] == ConversationalIntent.SOCIAL_CHECK_IN
    assert "functioning well" in r_soc["final_response"]

    # 5. Proposition
    r_prop = session.process_input("Ram lives in Delhi.")
    assert r_prop["intent"] == ConversationalIntent.PROPOSITION
    assert "Got it. Ram lives in Delhi." in r_prop["final_response"]


def test_canonical_baseline_intact():
    """Verify that language realization layers leave canonical baseline bit-exact."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    verif = verify_canonical_baseline(model)
    assert verif["total_params"] == CANONICAL_PARAM_COUNT
    assert verif["digest"] == CANONICAL_BASELINE_SHA256
    assert verif["delta_w"] == 0
    assert verif["is_bit_exact"] is True
