# CHAKRVIEW — CAND_01 EVALUATOR SPECIFICATION v1.1 (REVIEWED)
## Deterministic Metric Computation, Target Extraction, and Scoring Safeguards

**Document ID:** `DOC-EXP-CAND01-EVL-002`  
**Location:** `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.1_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 9  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope

This document represents the formal v1.1 scientific revision of `CAND_01_EVALUATOR_SPECIFICATION_v1.0.md`. It resolves critical and high-severity issues identified during independent evaluation harness review:
- **Correction 1 (First-Token vs. Full-Completion Distinction):** Explicitly distinguished `first_token_exact_match` from full-sequence completion accuracy and EOS sequence termination behavior.
- **Correction 2 (Comprehensive Metric Taxonomy):** Added explicit metrics for target-token logit rank, target-token pre-softmax probability, output length validity, and immediate EOS collapse.
- **Correction 3 (Negative Controls & Unit-Test Mandate):** Specified formal negative controls (empty context, wrong key query, key-shuffled context) and mandated automated evaluator unit tests prior to benchmark execution.
- **Correction 4 (Evaluator Verification Standards):** Mandated SHA-256 locking of the evaluator script and clarified that the 50-sample spot check is a qualitative audit, not mathematical proof of evaluator correctness.

---

## 2. Evaluator Architecture and Core Directives

The evaluation harness for CAND_01 is engineered to eliminate scoring ambiguity, false positives from unanchored substring searches, and false negatives from delimiter mismatches.

### Mandatory Operational Directives
1. **Zero Unanchored Substring Searching:**
   Constructs like `if target in completion:` are **strictly prohibited**. In small vocabulary distributions, short strings can appear by random chance or in repetitive loops, leading to inflated accuracy.
2. **Explicit First-Token vs. Full-Sequence Parsing:**
   Because all single-token prompts end with the canonical query delimiter ` -> `, the expected completion begins immediately at position 0 of the model's generated sequence.
   - **First-Token Exact Match ($A_{\text{FT}}$):** Compares `generated_tokens[0] == target_token_id`.
   - **Full Completion Match ($A_{\text{FC}}$):** Verifies that `generated_tokens[0] == target_token_id` **and** that the second token is `<EOS>` (`generated_tokens[1] == eos_id`), confirming clean single-token emission without trailing junk.
3. **Deterministic Greedy Decoding:**
   All benchmark evaluations are performed under greedy decoding (`temperature=0.0`, `top_k=1`, `top_p=1.0`) to ensure 100% bit-exact reproducibility across runs.

---

## 3. Metric Taxonomy and Mathematical Formulations

### 3.1 Primary Metric: First-Token Exact Match Accuracy ($A_{\text{FT}}$)
The proportion of evaluation instances where the first token emitted by the model exactly matches the target value token ID:
$$A_{\text{FT}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{t}_{i,0} = y_{i}^*\right)$$
Where $\hat{t}_{i,0}$ is the first generated token ID and $y_i^*$ is the ground-truth target token ID.

### 3.2 Secondary Metric: Full-Sequence Exact Match with Clean Termination ($A_{\text{FC}}$)
The proportion of instances where the model emits the target token followed immediately by `<EOS>`:
$$A_{\text{FC}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{t}_{i,0} = y_{i}^* \land \hat{t}_{i,1} = \text{eos\_id}\right)$$

### 3.3 Secondary Metric: Target Token Probability Rank ($R_{\text{tgt}}$)
Measures the rank of the correct target token in the model's pre-softmax logit distribution at the first response position (evaluated under teacher-forced prompt conditioning):
$$R_{\text{tgt}} = 1 + \sum_{v \in \mathcal{V}} \mathbb{I}\left(z_v > z_{y^*}\right)$$
- Optimal value: $R = 1$ (target token has highest logit).
- Empirical chance baseline: measured via key-shuffled controls.

### 3.4 Secondary Metric: Prompt Sensitivity Delta ($\Delta p$)
Measures the causal influence of prompt context on output probability by swapping the target value in context ($y_A \to y_B$) while keeping the query key unchanged:
$$\Delta p = \left| P(y_A \mid \text{Prompt}_{A}) - P(y_A \mid \text{Prompt}_{B}) \right|$$
- Target threshold: $\Delta p \ge 0.20$ (indicating strong causal coupling).
- Historical M5.1 failure benchmark: $\Delta p = 0.00014$.

---

## 4. Granular Error Classification Taxonomy

Every failed test instance ($\hat{t}_0 \neq y^*$) is automatically categorized into one of six mutually exclusive error classes:

| Error Code | Classification Name | Definition / Empirical Pattern | Interpretation |
|---|---|---|---|
| **ERR-01** | Distractor Intrusion | $\hat{t}_0 \in \{\text{Distractor Value Token IDs}\}$ | Model retrieved an irrelevant context value; attention routing failure. |
| **ERR-02** | Key Echo | $\hat{t}_0 \in \{\text{Query Key or Distractor Key IDs}\}$ | Model copied the key instead of the bound value; binding breakdown. |
| **ERR-03** | Syntax Echo | $\hat{t}_0 \in \{\text{"PAIR", "QUERY", "PAIRS", "COPY", "->" IDs}\}$ | Attention collapsed onto syntax prompt tokens; template attractor. |
| **ERR-04** | Repetition Loop | $\hat{t}$ exhibits repeating n-gram cycles | Softmax collapse into stationary cycle states. |
| **ERR-05** | Premature Termination | $\hat{t}_0 = \text{eos\_id}$ | Immediate `<EOS>` emission; failure to generate answer. |
| **ERR-06** | Out-of-Context Prior | $\hat{t}_0 \notin \{\text{All Context Token IDs}\}$ | Unanchored language model prior emission; hallucination. |

---

## 5. Mandatory Negative Controls & Validation Suite

Before an evaluation run is accepted, the evaluator must execute against three negative controls:

1. **Empty-Context Negative Control:**
   Present prompt `QUERY: [KEY] ->` with zero context pairs.
   - *Expected Behavior:* Model must not output target value at higher than baseline prior probability.
2. **Wrong-Key Negative Control:**
   Present prompt `PAIR: K1=V1. QUERY: K2 ->` where $K_2$ was never defined.
   - *Expected Behavior:* Model must not emit $V_1$ ($ERR-01$ avoidance).
3. **Shuffled-Mapping Permutation Control:**
   Present prompt with all keys and values permuted across examples.
   - *Expected Behavior:* Measures background associative rate.

### 5.1 Evaluator Unit-Test Specification
The prospective evaluator script (`scripts/evaluate_cand01_benchmark.py`) must pass a dedicated unit-test suite (`tests/test_cand01_evaluator.py`) verifying that:
- It correctly flags string substrings as non-matches when token IDs differ.
- It correctly penalizes extra trailing tokens in $A_{\text{FC}}$.
- It correctly scores empty generation as $ERR-05$.
- It raises an error if vocabulary size or tokenizer artifacts mismatch.

---

## 6. Audit Protocol & Statistical Aggregation

### 6.1 Multi-Seed Aggregation
The evaluator script must process all three experimental seeds (`seeds: [42, 137, 2026]`) and output:
- Per-seed exact match scores.
- Mean, standard deviation, and standard error across seeds.
- 95% Wilson score confidence intervals for each arm.
- Statistical significance tests (paired t-test / Wilcoxon) comparing Arm A against Arms B, C, and D.

### 6.2 Qualitative 50-Sample Spot Check
- A pseudo-random sample of 50 evaluation instances (`seed=999`) will be inspected by an independent reviewer.
- **Protocol Clarification:** This manual spot-check serves as a **qualitative sanity audit** to detect logging formatting defects; it does **not** substitute for formal unit-test validation or mathematical proof of evaluator correctness.

---

## 7. Evaluator Governance & Artifact Locking

- **Script Identifier (Prospective):** `scripts/evaluate_cand01_benchmark.py`
- **Benchmark Suite Identifier (Prospective):** `experiments/cand01_controlled_entity_copying/benchmark/cand01_test_benchmark_500.json`
- **Locking Requirement:** The evaluator script and benchmark file must have their SHA-256 hashes recorded in the experiment completion manifest before post-training evaluation begins.
- **Execution Mandate:** This document specifies evaluation methodology only. **No evaluator execution is authorized by this document.**

