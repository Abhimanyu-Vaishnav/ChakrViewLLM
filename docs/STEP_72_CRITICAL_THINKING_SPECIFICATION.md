# ChakrView Step 72 Specification: Critical Thinking & Alternative Hypotheses

## 1. Architectural Purpose
Step 72 extends the reasoning subsystem with explicit critical evaluation capabilities. Rather than blindly accepting a single generated candidate direction, the [CriticalThinkingEngine](file:///d:/Project/ChakrView/chakrview/cognition/reasoning/critical.py) audits evidence quality, detects contradictions, identifies vulnerable assumptions, and explicitly formulates alternative hypotheses.

## 2. Core Concepts

### 2.1 Evidence Balance
For every claim, the engine determines:
- Corroborating evidence vs contradictory evidence.
- `EvidenceStrength`: `DEFINITIVE`, `CORROBORATED`, `INDIRECT`, `CONTESTED`, `STALE`, or `INSUFFICIENT`.
- Stale state detection relative to current repository facts.

### 2.2 Alternative Hypotheses
To prevent prematurely narrowing on a potentially flawed plan, the engine synthesizes competing explanations and implementation options:
- `AlternativeHypothesis`: Explicit description, plausibility score, required evidence, and architectural tradeoffs.

### 2.3 Critical Analysis Report
The audit result provides:
- Breakdown of all claim evidence balances.
- List of competing alternative hypotheses.
- Vulnerable unverified assumptions.
- Explicit list of unresolved contradictions.
- Epistemic reliability score ($0.0 \le S \le 1.0$).
- Actionable recommendation: `PROCEED`, `EXPLORE_ALTERNATIVES`, `ACQUIRE_EVIDENCE`, or `ABSTAIN`.

## 3. Epistemic Invariants
- Contradictory evidence is never suppressed or ignored.
- If negative boundaries or quarantined context items are referenced, the engine mandates `ABSTAIN`.
- When evidence is missing, the engine reports `INSUFFICIENT` rather than hallucinating corroboration.
