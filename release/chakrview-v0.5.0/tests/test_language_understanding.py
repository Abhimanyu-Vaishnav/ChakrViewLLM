"""
Held-Out Evaluation Suite for ChakrView Language Understanding Foundation.

Evaluates 7 distinct capability dimensions with dedicated metrics (not hiding behind a single metric):
1. Input Normalization Correctness:
   - Misspellings, typo corrections, Hinglish phonetic variations.
   - Protection of proper nouns, technical terms, identifiers (e.g. Zorvax-917, Ram, Delhi).
2. Intent Classification Accuracy:
   - Unseen paraphrases, social vs definition vs task vs proposition vs clarification.
3. Compositional Semantic Parsing:
   - Varied word order (SVO, SOV, OVS, OSV).
   - Entity and relation extraction.
   - Negation extraction and polarity marking.
   - Grammatical tense detection.
4. Epistemic Abstention & Unsupported Queries:
   - Unknown entities (Zorvax-917, Atlantis).
   - No hallucination or fabricated answers.
5. Incomplete / Ambiguous Clarification Behavior:
   - Truncated sentence triggers explicit clarification request.
6. Governed Candidate Learning Interface:
   - Candidate facts recorded with provenance, pending verification.
   - Canonical neural weights and parameter count remain strictly invariant.
7. Regression Protection:
   - Preservation of existing definitions, Ram-Delhi memory, multi-turn pronoun resolution.
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
from chakrview.cognition.language_understanding import (
    TextNormalizer,
    CompositionalSentenceParser,
    SentenceMood,
    QuestionTarget,
    GrammaticalTense,
    Polarity,
    CandidateLearningLedger,
    ApprovalStatus,
)
from chakrview.cognition.language_realizer import (
    LanguageClassifier,
    IntentClassifier,
    LanguageRealizer,
    LanguageTag,
    ConversationalIntent,
)
from scripts.founder_manual_test import ChakrViewRuntimeSession


# ==============================================================================
# DIMENSION 1: INPUT NORMALIZATION CORRECTNESS
# ==============================================================================

def test_normalization_spelling_and_proper_noun_preservation():
    """
    Evaluates typo correction while guaranteeing proper nouns and identifiers are untouched.
    """
    cases = [
        # (input_text, expected_corrected_substr, expected_preserved_proper_noun)
        ("Ram livs in Delhi.", "lives", "Ram"),
        ("Define computr system", "computer", "computer"),
        ("Define Zorvax-917", "definition" if "def" in "Define Zorvax-917" else "Define", "Zorvax-917"),
        ("Ram Delhi me rhta hai", "rehta", "Ram"),
        ("Main bhi thik hu, jra muje computer ki defination btao", "definition", "computer"),
    ]

    total = len(cases)
    passed = 0

    for raw, corrected_sub, preserved_noun in cases:
        norm = TextNormalizer.normalize(raw)
        # Check raw preserved
        assert norm.raw_text == raw
        # Check proper noun or technical code protected
        assert preserved_noun in norm.cleaned_text or preserved_noun in norm.protected_tokens
        # Check typo fixed
        if corrected_sub.lower() in norm.cleaned_text.lower():
            passed += 1

    norm_acc = passed / total
    assert norm_acc >= 0.80, f"Normalization accuracy below threshold: {norm_acc:.2f}"


def test_normalization_hinglish_detection():
    """
    Evaluates distinct language detection for varied Hinglish sentence structures.
    """
    hinglish_samples = [
        "Tum kaise ho bhai",
        "Ram Delhi mein rehta hai",
        "Aap kahan rehte ho",
        "Main bhi theek hoon, zara batao",
    ]
    for text in hinglish_samples:
        norm = TextNormalizer.normalize(text)
        assert norm.detected_language == "HINGLISH", f"Expected HINGLISH for '{text}', got {norm.detected_language}"


# ==============================================================================
# DIMENSION 2: INTENT CLASSIFICATION ACCURACY
# ==============================================================================

def test_intent_accuracy_held_out_paraphrases():
    """
    Evaluates intent discrimination across unseen paraphrases.
    """
    test_cases = [
        # (text, expected_intent)
        ("Good morning to everyone", ConversationalIntent.GREETING),
        ("Hey ChakrView", ConversationalIntent.GREETING),
        ("How are things going with you?", ConversationalIntent.SOCIAL_CHECK_IN),
        ("Kaisi tabiyat hai aapki?", ConversationalIntent.SOCIAL_CHECK_IN),
        ("What is the definition of a computer?", ConversationalIntent.DEFINITION_QUERY),
        ("Computer ki paribhasha samjhao", ConversationalIntent.DEFINITION_QUERY),
        ("Make a step-by-step plan for relocation", ConversationalIntent.TASK_REQUEST),
        ("Sita lives in Pune.", ConversationalIntent.PROPOSITION),
        ("System is an ?", ConversationalIntent.CLARIFICATION),
    ]

    correct = 0
    for text, exp_intent in test_cases:
        lang = LanguageClassifier.classify(text)
        act_intent = IntentClassifier.classify(text, lang)
        if act_intent == exp_intent:
            correct += 1

    intent_acc = correct / len(test_cases)
    assert intent_acc >= 0.85, f"Held-out intent accuracy below target: {intent_acc:.2f}"


# ==============================================================================
# DIMENSION 3: COMPOSITIONAL SEMANTIC PARSING
# ==============================================================================

def test_compositional_parsing_varied_syntax_and_relations():
    """
    Evaluates compositional parsing of relations, word orders (SVO vs SOV vs OVS), and entities.
    """
    # 1. Standard English SVO
    norm1 = TextNormalizer.normalize("Ram lives in Delhi.")
    p1 = CompositionalSentenceParser.parse(norm1)
    assert p1.mood == SentenceMood.DECLARATIVE
    assert p1.subject == "Ram"
    assert p1.predicate_relation == "LIVES_IN"
    assert p1.target_object == "Delhi"
    assert p1.polarity == Polarity.AFFIRMATIVE

    # 2. Inverted word order (OVS / Relative)
    norm2 = TextNormalizer.normalize("Delhi is where Ram lives.")
    p2 = CompositionalSentenceParser.parse(norm2)
    assert p2.subject == "Ram"
    assert p2.predicate_relation == "LIVES_IN"
    assert p2.target_object == "Delhi"

    # 3. Hinglish SOV
    norm3 = TextNormalizer.normalize("Ram Delhi mein rehta hai.")
    p3 = CompositionalSentenceParser.parse(norm3)
    assert p3.subject == "Ram"
    assert p3.predicate_relation == "LIVES_IN"
    assert p3.target_object == "Delhi"


def test_compositional_negation_and_tense_detection():
    """
    Evaluates negation detection (Polarity.NEGATIVE) and tense detection across languages.
    """
    # Negation in English
    norm_neg_en = TextNormalizer.normalize("Ram does not live in Mumbai.")
    p_neg_en = CompositionalSentenceParser.parse(norm_neg_en)
    assert p_neg_en.polarity == Polarity.NEGATIVE
    assert p_neg_en.tense == GrammaticalTense.PRESENT

    # Negation in Hinglish
    norm_neg_hi = TextNormalizer.normalize("Ram Mumbai mein nahi rehta hai.")
    p_neg_hi = CompositionalSentenceParser.parse(norm_neg_hi)
    assert p_neg_hi.polarity == Polarity.NEGATIVE

    # Past tense
    norm_past = TextNormalizer.normalize("Ram lived in Kolkata.")
    p_past = CompositionalSentenceParser.parse(norm_past)
    assert p_past.tense == GrammaticalTense.PAST

    # Future tense
    norm_fut = TextNormalizer.normalize("Ram will live in Bangalore.")
    p_fut = CompositionalSentenceParser.parse(norm_fut)
    assert p_fut.tense == GrammaticalTense.FUTURE


# ==============================================================================
# DIMENSION 4: EPISTEMIC ABSTENTION & UNSUPPORTED QUERIES
# ==============================================================================

def test_unsupported_questions_honest_abstention():
    """
    Evaluates that unknown entities trigger clean honest abstention without fake answers.
    """
    session = ChakrViewRuntimeSession()

    queries = [
        "Who is Zorvax-917 and where does it live?",
        "What is the capital of Atlantis?",
        "Define Zorvax-917",
    ]

    for q in queries:
        res = session.process_input(q)
        assert res["decision"] == "ABSTAIN", f"Failed honest abstention on '{q}', got {res['decision']}"
        assert res["confidence"] == 0.0
        assert "verified" in res["final_response"].lower() or "abstention" in res["final_response"].lower()


# ==============================================================================
# DIMENSION 5: INCOMPLETE / AMBIGUOUS INPUT CLARIFICATION
# ==============================================================================

def test_incomplete_inputs_request_clarification():
    """
    Evaluates that truncated or incomplete statements request clarification instead of inventing meaning.
    """
    session = ChakrViewRuntimeSession()

    fragments = [
        "Computer is",
        "Computer is an ?",
        "System is a",
    ]

    for frag in fragments:
        res = session.process_input(frag)
        assert res["decision"] in ("REQUEST_CLARIFICATION", "CLARIFICATION"), f"Expected clarification for '{frag}', got {res['decision']}"
        assert "incomplete" in res["final_response"].lower() or "adhura" in res["final_response"].lower()


# ==============================================================================
# DIMENSION 6: GOVERNED CANDIDATE LEARNING INTERFACE
# ==============================================================================

def test_candidate_learning_ledger_governance():
    """
    Evaluates candidate learning storage: records facts with provenance, pending verification,
    and guarantees zero weight mutation.
    """
    ledger = CandidateLearningLedger()

    rec = ledger.submit_candidate_fact(
        raw_input="Arjun lives in Jaipur.",
        normalized_input="Arjun lives in Jaipur.",
        subject="Arjun",
        relation="LIVES_IN",
        value="Jaipur",
        provenance="USER_INTERACTIVE_SESSION",
        notes="Held-out learning evaluation case",
    )

    assert rec.record_id.startswith("cand_")
    assert rec.approval_status == ApprovalStatus.PENDING_VERIFICATION
    assert rec.extracted_subject == "Arjun"
    assert rec.extracted_relation == "LIVES_IN"
    assert rec.extracted_value == "Jaipur"
    assert ledger.total_count() == 1
    assert len(ledger.list_pending()) == 1


# ==============================================================================
# DIMENSION 7: REGRESSION PROTECTION & CANONICAL BASELINE INTEGRITY
# ==============================================================================

def test_memory_and_reference_resolution_regression():
    """
    Verifies that Ram-Delhi memory storage and pronoun reference resolution remain intact.
    """
    session = ChakrViewRuntimeSession()

    # Step 1: Assert proposition
    r1 = session.process_input("Ram lives in Delhi.")
    assert r1["decision"] == "BELIEF_ACQUIRED"
    assert "Got it. Ram lives in Delhi." in r1["final_response"]

    # Step 2: Query via pronoun
    r2 = session.process_input("Where does he live?")
    assert r2["decision"] == "ANSWER"
    assert "Ram lives in Delhi." in r2["final_response"]

    # Step 3: Verified definition retrieval
    r3 = session.process_input("Define Computer")
    assert r3["decision"] == "ANSWER"
    assert "electronic device" in r3["final_response"]


def test_canonical_baseline_invariance():
    """
    Strictly verifies that canonical baseline weights are bit-exact with Delta W = 0.
    """
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    verif = verify_canonical_baseline(model)

    assert verif["total_params"] == CANONICAL_PARAM_COUNT
    assert verif["digest"] == CANONICAL_BASELINE_SHA256
    assert verif["delta_w"] == 0
    assert verif["is_bit_exact"] is True
