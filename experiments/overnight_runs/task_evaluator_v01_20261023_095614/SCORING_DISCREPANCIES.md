# Scoring Discrepancies & Audit Reconciliation Report

**Audit Scope**: Exp17 Held-out Test Split (40 records) & Normal Chat Gate v0.1 (20 prompts)

**Evaluator**: TaskEvaluator v0.1 (chakrview.evaluation.task_evaluator)

**Date**: 2026-10-23

---

## 1. Executive Summary

Prior validation and mission reports reported a dramatic improvement in model behavior, specifically citing an EOS turn-termination rate of 45.0% for SFT Step 225 vs 0.0% for Baseline, alongside 100% EOS in structured formatting and 80% EOS in factual QA.

When evaluated using TaskEvaluator v0.1, which implements deterministic correctness checks rather than turn-termination flags, we observe that:

- Baseline Scorable Task Accuracy: 0.0% (0 / 35)

- SFT Step 225 Scorable Task Accuracy: 0.0% (0 / 35)

- Scorable Coverage: 87.5% (35 / 40) (5 short explanations held out for human review)

This audit establishes definitively that conversational turn-taking was acquired, but task reasoning capability was not. Every claim conflating EOS emissions or response formatting with actual factual/reasoning correctness is hereby corrected.

---

## 2. Category-by-Category Discrepancy Analysis

### 2.1. Structured Formatting (structured_formatting)

- Previous Claim: Step 225 achieved 100% (5/5) EOS termination. Prior summaries informally characterized this as formatting success.

- Rescored Reality: Syntax Validity: 0.0% (0 / 5). Schema Validity: 0.0% (0 / 5).

- In 2 records (EXP17_TEST_STRUCTURED_FORMATTING_000 and 001), the model generated an empty response before stopping.

- In 3 records (002, 003, 004), the model emitted Python docstring/code syntax fragments.

- RFC 8259 JSON parser (json.loads) correctly rejects 100% of these outputs as JSON_SYNTAX_INVALID.

### 2.2. Basic Arithmetic (basic_arithmetic)

- Previous Claim: Step 225 achieved 60.0% (3/5) EOS termination, generating sentences that mimicked arithmetic answers.

- Rescored Reality: Task Accuracy: 0.0% (0 / 5).

- Example EXP17_TEST_BASIC_ARITHMETIC_000: Prompt asks for sum of 503 and 505 (expected: 1008). Model generated: The sum of 2005 is 406.

- Example EXP17_TEST_BASIC_ARITHMETIC_001: Prompt asks for product of 507 and 2 (expected: 1014). Model generated: The primary of 21 is 406.

- Word boundary regex validation rejected all responses. The model learned the superficial surface form of a mathematical sentence without acquiring arithmetic computation.

### 2.3. Uncertainty and Abstention (uncertainty_abstention)

- Previous Claim: 100% semantic pass in dataset audit.

- Rescored Reality: Task Accuracy: 0.0% (0 / 5).

- On the 4 unanswerable test items, the model failed to emit recognized refusal phrases, instead babbling or emitting incomplete fragments.

- On the 1 answerable control item (EXP17_TEST_UNCERTAINTY_ABSTENTION_004), the expected answer was Doctor of Philosophy, but the model generated unrelated tokens.

### 2.4. Context Extraction & Reference Resolution

- Rescored Reality: Task Accuracy: 0.0% (0 / 5) for Context Extraction; 0.0% (0 / 5) for Reference Resolution.

- Although the model reduced empty-string outputs (down to 0% from 80% baseline), its generated sequences did not extract the grounded focal entity.

### 2.5. Short Explanations (short_explanation)

- Previous Claim: Semantic audit claimed 100% PASS based on a hardcoded test script assigning PASS.

- Rescored Reality: Evaluator assigned NEEDS_REVIEW (UNSCORED) to all 5 items.

- Offline deterministic keyword matching cannot verify causal scientific mechanisms without unacceptable false positives. Honest evaluation requires reporting scorable coverage as 35/40 rather than fabricating a pass/fail score.

---

## 3. Normal Chat Gate v0.1 Discrepancy & Verification

On the frozen 20-prompt Normal Chat Gate v0.1 suite:

- EOS Turn Termination: Baseline: 5.0% (1/20) -> SFT Step 225: 60.0% (12/20).

- Max-Tokens Truncation: Baseline: 95.0% (19/20) -> SFT Step 225: 40.0% (8/20).

- Empty Output Rate: Baseline: 0.0% (0/20) -> SFT Step 225: 45.0% (9/20).

- Trigram Repetition Rate: Baseline: 46.2% -> SFT Step 225: 13.3%.

Takeaway: SFT sharply suppressed degenerate repetition loops and learned turn-ending behavior, but frequently over-compensated by stopping immediately (yielding empty outputs) or failing to answer the prompt factually.