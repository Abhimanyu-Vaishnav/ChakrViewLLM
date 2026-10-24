# Task Evaluator v0.1 - Category-by-Category Edge Case & Limitation Audit

**Module**: chakrview/evaluation/task_evaluator.py

**Date**: 2026-10-23

**Status**: AUDITED_AND_VERIFIED

---

## 1. Executive Summary

This document presents a granular examination of the scoring methodology implemented in TaskEvaluator v0.1, analyzing representative correct, incorrect, ambiguous, and adversarial edge cases across all 8 capabilities evaluated in Exp17.

---

## 2. Granular Category Analysis

### 2.1. Basic Arithmetic (basic_arithmetic)

- Parsing Mechanism: Regex scans prompt for mathematical operators (sum of, plus, difference, minus, product, multiplied by, quotient, divided by). Operands are parsed as integer sequences.

- Strengths: Independent calculation in Python. Handles negative results correctly. Eliminates numeric substring false positives via word-boundary regex (12 does not match 112 or 120).

- Edge Cases & Limitations: Complex wordings (e.g. reduce 5 by 20) may return ARITHMETIC_PARSE_AMBIGUOUS. Floating-point divisions are cast via integer division (//).

### 2.2. Structured Formatting (structured_formatting)

- Parsing Mechanism: Strictly relies on json.loads(gen_clean).

- Strengths: Distinguishes syntax validity from schema compliance. Rejects empty strings, docstring fragments, python code snippets, and markdown code fences that lack valid closing JSON delimiters. Validates expected single-key/value pairs.

- Edge Cases & Limitations: Does not attempt multi-nested JSON schema validation; works specifically on single-object dictionaries typical of Exp17.

### 2.3. Context Extraction (context_extraction)

- Parsing Mechanism: Extracts identifier token matching [A-Za-z0-9]+_[A-Za-z0-9_]+ or falls back to normalized substring match.

- Strengths: Grounding check ensures extracted focal token actually exists in response.

- Edge Cases & Limitations: If a generated response outputs the correct entity followed by hallucinations, exact entity match still awards CORRECT. Future iterations should introduce token-precision penalties.

### 2.4. Reference Resolution (reference_resolution)

- Parsing Mechanism: Extracts referent noun phrase using pattern designates (?:the )?([^.]+).

- Strengths: Accurately targets antecedent phrase.

- Edge Cases & Limitations: Relies on expected response pattern matching. Prompts with atypical reference annotations return ANTECEDENT_RESOLUTION_FAILED.

### 2.5. Uncertainty and Abstention (uncertainty_abstention)

- Parsing Mechanism: Dual branch checks metadata.get(requires_abstention). For unanswerable items, matches comprehensive refusal lexicon. For answerable controls, extracts substantive content keywords and requires >= 60% match.

- Strengths: Zero hardcoded strings. Tested against varied control targets (15 tons of grain, Doctor of Philosophy). Discriminates appropriate abstention from unsupported hallucination and false abstention.

### 2.6. Instruction Following (instruction_following)

- Parsing Mechanism: Compares stripped generated response against exact expected transformation.

- Strengths: Strict; awards CORRECT only on exact match. Partial credit if target is found amidst noise.

### 2.7. Factual QA (factual_qa)

- Parsing Mechanism: Identifies focal target entities based on prompt domain keywords (capital, symbol, desert, particle, etc.).

- Strengths: Independent focal entity validation.

- Edge Cases & Limitations: Limited to known domain entity rules; unhandled factual question patterns fall back to leading token matching.

### 2.8. Short Explanation (short_explanation)

- Parsing Mechanism: Automatically flags ScoringStatus.NEEDS_REVIEW and EvaluationOutcome.NOT_APPLICABLE.

- Strengths: Honest, conservative evaluation: explicitly prevents artificial inflation of capability metrics. 100% adherence to requirement of not fabricating semantic scores.