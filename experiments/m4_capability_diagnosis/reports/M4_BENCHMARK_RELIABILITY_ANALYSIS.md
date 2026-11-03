# CHAKRVIEW — MILESTONE M4 BENCHMARK RELIABILITY ANALYSIS
## Systematic Audit of Evaluators, Scoring Rules, Sensitivity & Measurement Validity

**Project:** ChakrView  
**Milestone:** M4 — Capability Diagnosis & Learning Strategy Research  
**Evaluators Inspected:** `scripts/run_m3_evaluations.py`, `evaluation/intelligence_suite/scoring_evaluator.py`, `evaluation/coding_suite/run_coding_evaluation.py`, `evaluation/neural_benchmark_harness.py`  
**Date:** 2026-10-30  

---

## 1. SUITE-BY-SUITE CAPABILITY MAPPING

| Benchmark Suite | Task Count | Capability Target | Scoring Method | Primary Vulnerability |
|---|---|---|---|---|
| **M3 Conversational** | 60 | Dialogue, instructions, context, multi-turn, arithmetic | Rule-based regex & substring | Substring false positives; rigid target wording |
| **Intelligence Suite** | 32 | Native language, coding, reasoning, format | Rule-based (`exact_strip`, `word_boundary`, `code_functional`) | Fallback to AST syntax validation produces false positives |
| **Coding Suite v0.7** | 12 | Python syntax and unit test execution | Process-isolated executor | Security policy rejects un-executed code (6/12 rejected) |
| **Neural Harness** | 30 | Generation dynamics (EOS, repetition, token diversity, TPS) | Statistical n-gram & unique token tracking | High reliability; measures mechanical dynamics, not semantics |

---

## 2. EVALUATOR AUDIT & VULNERABILITIES

### 2.1 The `code_functional` Fallback False Positive (Audit E Root Cause)
In `evaluation/intelligence_suite/scoring_evaluator.py`:
```python
elif scoring_rule == "code_functional":
    code_to_eval = (full_prompt + clean_gen) if full_prompt else clean_gen
    if test_code and ISOLATED_EXECUTOR_AVAILABLE:
        # runs unit tests
        ...
    else:
        # Fall back to syntax validity check
        try:
            ast.parse(code_to_eval)
            return True, "SYNTAX_VALID"
        except SyntaxError as e:
            return False, f"SYNTAX_ERROR: {e.msg}"
```
**Forensic Defect Identified:**
On task `CODE-03` (`# Python assertion checking that len('abc') equals 3:\nassert `), no `test_code` was defined in `intelligence_tasks.json`. Therefore, the evaluator fell back to `ast.parse()`.
- When M3 Step 250 generated `dp.`, the full string was `assert dp.`, which is a valid attribute access assertion under AST syntax.
- The model did not complete `len('abc') == 3`. It produced a random token `dp.` that happened to be syntactically legal Python.
- This resulted in an artificial "pass" for Canonical Step 10k, Step 250, and Step 500.

### 2.2 Substring Matching False Positives in Conversational Benchmark
In `evaluation/conversational_suite/m3_conversational_tasks.json`, task `M3-EVAL-GQA-04`:
- Prompt: `"Context: In Python, tuples are immutable while lists are mutable.\nQuestion: Can a tuple be modified after its creation?\nAssistant:"`
- Target: `"No, tuples cannot be modified because they are immutable."`
- Expected concept: `"No"`
- Scoring Rule: `"substring"`
- Model Generation (Step 250): `"The largest cannot have a non-positive face."`
- Evaluator Result: **PASSED**.
- Why? Because `"non-positive"` contains the lowercase substring `"no"`.
- **Verdict:** The single conversational task passed by Step 250 (1/60) was a **complete false positive**. Real conversational pass rate across all models is **0/60 (0.0%)**.

### 2.3 Prompt Format Sensitivity
- In M2, prompts were processed through `NeuralRuntimeAdapter`, which called `.strip()`. For trailing prompts like `"assert "`, stripping removed the space, creating `"assert"`.
- In M3, `DirectNeuralRunner` passed raw strings directly, preserving the space.
- This single-space difference changed BPE tokenization from token `['assert']` to tokens `['assert', ' ']`, altering the greedy decoding trajectory across 9 of 32 tasks.

---

## 3. SEPARATION OF FORM VERSUS SUBSTANCE

The existing benchmark reports must be disaggregated into mechanical vs semantic success:

1. **Mechanical Success (VERIFIED):**
   - Repetition collapse is authentically prevented: harness 4-gram repetition dropped from 0.2540 to 0.0223.
   - EOS termination is authentically learned: from 6.7% to 93.3%.
   - Token diversity is authentically high: unique token ratio increased from 0.5314 to 0.9036.
2. **Semantic Capability (FAILED / ZERO EVIDENCE):**
   - True conversational question answering: **0 / 60 (0.0%)** (after correcting false positive).
   - Multi-turn state tracking: **0 / 10 (0.0%)**.
   - Functional coding execution: **0 / 12 (0.0%)**.
   - Arithmetic calculation: **0 / 5 (0.0%)**.

---

## 4. EVALUATOR REPAIR RECOMMENDATIONS FOR M5

1. **Eliminate Syntax-Only Fallbacks for Assertion Tasks:** Require functional unit tests or exact AST match for code tasks; do not award "pass" to incomplete fragments.
2. **Word-Boundary Match over Substring Match:** For single-word concepts like `"no"` or `"ready"`, strictly require word boundaries (`\bno\b`) to eliminate false substring positives like `"non-positive"`.
3. **Rigid Prompt-Boundary Invariant:** Require all runners (adapter, pipeline, standalone) to use an identical tokenizer contract for BOS, EOS, and whitespace handling.

