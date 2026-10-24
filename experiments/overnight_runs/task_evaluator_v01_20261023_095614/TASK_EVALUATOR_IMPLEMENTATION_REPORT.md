# Task Evaluator v0.1 - Implementation Report

**Module**: chakrview/evaluation/task_evaluator.py

**Package**: chakrview.evaluation

**Author**: Senior ML Evaluation Engineer

**Date**: 2026-10-23

**Status**: IMPLEMENTED_AND_VERIFIED

---

## 1. Executive Summary & Objective

In accordance with the ChakrView Exp17 evaluation mandate, a reliable, offline, deterministic Task Evaluator (TaskEvaluator v0.1) has been designed and implemented in chakrview/evaluation/task_evaluator.py.

The primary objective of TaskEvaluator is to provide ground-truth-anchored, deterministic task scoring across all 8 capabilities present in the Exp17 Instruction SFT suite, completely separating:

1. Conversational turn termination (EOS%) from semantic task correctness (Task Accuracy%).

2. Format validity (e.g., valid RFC 8259 JSON) from raw string presence.

3. Scorable coverage (records with unambiguous deterministic targets) from subjective or unverifiable records.

The evaluator operates 100% offline with zero external network dependencies, no external pretrained models, no non-deterministic heuristic scores, and strictly rejects converting unknown or unparseable outputs into passing scores.

---

## 2. Architecture & Public Interface

The module exports four core classes/enums:

- ScoringStatus (Enum): SCORED, UNSCORED, NEEDS_REVIEW, ERROR.

- EvaluationOutcome (Enum): CORRECT, INCORRECT, PARTIALLY_CORRECT, NOT_APPLICABLE.

- EvaluationResult (Dataclass): record_id, category, status, outcome, expected_target, observed_evidence, reason_codes, evaluator_version, limitations.

- TaskEvaluator (Class): evaluate_record(record, generated_text), evaluate_batch(records, generated_texts).

---

## 3. Category-Specific Deterministic Scoring Rubrics

### 3.1. Basic Arithmetic (basic_arithmetic)

- Derivation: Parses mathematical operations (+, -, *, //) directly from prompt text.

- Independent Computation: Computes expected integer result independently in Python (supporting negative integers and integer divisions).

- False-Positive Prevention: Compares the generated answer using word-boundary regular expressions, completely eliminating substring false positives such as matching 12 inside 112.

- Reason Codes: ARITH_CORRECT, ARITH_MISMATCH, ARITH_PROMPT_UNPARSED.

### 3.2. Structured Formatting (structured_formatting)

- Strict RFC 8259 Parsing: Evaluates complete response using standard JSON deserialization (json.loads).

- Syntax vs Schema Distinction: Rejects empty strings, docstrings, code fragments, markdown prose, or partial JSON strings with reason code JSON_SYNTAX_INVALID. If valid JSON syntax is parsed, validates schema constraints.

- Reason Codes: JSON_VALID_SCHEMA_MATCH, JSON_SCHEMA_MISMATCH, JSON_SYNTAX_INVALID.

### 3.3. Context Extraction (context_extraction)

- Target Derivation: Extracts ground-truth entity from record expected answer and confirms presence in supplied context.

- Normalized Exact Match: Normalizes whitespace, casing, and punctuation. Checks if normalized target entity is present as exact token/phrase in response.

- Grounding vs Hallucination: Verifies model generated answer contains the specific focal entity rather than generic context tokens.

- Reason Codes: EXTRACTION_CORRECT, EXTRACTION_INCORRECT, TARGET_UNIDENTIFIED.

### 3.4. Reference Resolution (reference_resolution)

- Antecedent Grounding: Identifies specific antecedent or referent from target field.

- Ambiguity Detection: If context presents contradictory referents or ambiguous pronoun bindings, marks ScoringStatus.UNSCORED with AMBIGUOUS_REFERENCE.

- Reason Codes: REF_RESOLVED, REF_MISMATCH, REF_AMBIGUOUS.

### 3.5. Uncertainty and Abstention (uncertainty_abstention)

- Independent Verification: Does not blindly rely on requires_abstention boolean flag. Checks whether requested attribute is factually present in context.

- Dual Outcome Tracking: Unanswerable prompts check for appropriate refusal phrases (I do not possess, The provided text does not contain, insufficient evidence, cannot determine). Answerable controls check for specific target fact dynamically extracted from each individual record expected answer. Zero hardcoded target strings.

- Reason Codes: CORRECT_ABSTENTION, UNSUPPORTED_ANSWER, GROUNDED_ANSWER, INCORRECT_ABSTENTION.

### 3.6. Instruction Following (instruction_following)

- Constraint Verification: Parses prompt operational command (uppercase, lowercase, string reversal, single-word formatting).

- Strict Verification: Evaluates whether generated text strictly satisfies the constraint.

- Reason Codes: INSTRUCTION_FOLLOWED, INSTRUCTION_VIOLATED.

### 3.7. Factual QA (factual_qa)

- Focal Entity Match: Uses independently established target entities and aliases (e.g. Canberra for capital of Australia, Mercury for smallest planet).

- Reason Codes: FACT_CORRECT, FACT_INCORRECT.

### 3.8. Short Explanation (short_explanation)

- Honest Handling: Categorical semantic explanations require human domain judgment or causal entailment that cannot be reliably reduced to keyword matching without fabricating scores.

- Status Assignment: Explicitly returns ScoringStatus.NEEDS_REVIEW and EvaluationOutcome.NOT_APPLICABLE with reason code EXPLANATION_NEEDS_HUMAN_REVIEW.

- Scorable Coverage Impact: Properly documented as unscored, adjusting total-suite scorable coverage to 87.5% (35/40).

---

## 4. Verification and Regression Testing

A dedicated test suite was implemented in tests/test_task_evaluator.py:

- 16 comprehensive test cases testing all 8 categories, boundary conditions, negative arithmetic, non-integer division, numeric substring collision avoidance, JSON docstring rejection, abstention controls with distinct targets, and invalid records.

- Execution Result: 16 / 16 PASSED in 0.006s (Exit code: 0).

- Full Repository Regressions: 122 / 122 PASSED in 118.29s (Exit code: 0).

- Exp17 Test Suite: 9 / 9 PASSED in 27.41s (Exit code: 0).

Total Test Suite Verdict: 147 / 147 PASSED - ZERO REGRESSIONS.