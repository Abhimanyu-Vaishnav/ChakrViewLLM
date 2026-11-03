# CHAKRVIEW — CAND_01 EVALUATOR SPECIFICATION v1.3 (REVIEWED)
## Deterministic Metric Computation, Target Extraction, and Scoring Safeguards

**Document ID:** `DOC-EXP-CAND01-EVL-004`  
**Location:** `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.2_REVIEWED.md`, `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 9  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope (v1.3 Targeted Redesign)

This document represents the formal v1.3 scientific revision of `CAND_01_EVALUATOR_SPECIFICATION_v1.2_REVIEWED.md`. It incorporates specific resolutions to independent audit findings (ADV-03, ADV-04, ADV-05, and ADV-07):
- **Correction ADV-07 (Two-Way Clustered Uncertainty):** Replaced target-only clustering with a **two-way cluster bootstrap** resampled simultaneously over unique key token IDs and unique target token IDs ($B=2,000$ resamples), preventing underestimation of variance from key embedding degeneration.
- **Correction ADV-03 & ADV-04 (Statistical Test Realism & Model Simplification):** Removed the mathematically unachievable paired seed permutation test ($p_{\text{min}} = 0.25$ for $n=3$). Formulated mixed-effects logistic regression strictly with crossed random intercepts for keys and values ($u_{\text{key}}$ and $w_{\text{val}}$), removing the unidentifiable prompt random effect.
- **Correction ADV-08 (Dual Reporting Enforcement):** Strictly enforced that Recombination ($A_{\text{test,recomb}}$) and Disjoint-Token In-Context Copying ($A_{\text{test,disjoint}}$) are reported as separate primary benchmarks and never pooled into a single composite metric.

---

## 2. Core Evaluator Directives & Scoring Safeguards

### Mandatory Operational Directives
1. **Zero Unanchored Substring Searching:**
   Constructs like `if target in completion:` are **strictly prohibited**. In small vocabulary distributions, short strings can appear by random chance or in repetitive loops, leading to inflated accuracy.
2. **Coupled First-Token and Clean Completion Scoring:**
   - **First-Token Exact Match ($A_{\text{FT}}$):** Compares `generated_tokens[0] == target_token_id`.
   - **Full Completion Match with Clean Termination ($A_{\text{FC}}$):** Verifies that `generated_tokens[0] == target_token_id` **and** `generated_tokens[1] == eos_id` **and** `len(generated_tokens) == 2`.
   - *Direct Policy:* Emitting the correct first token followed by trailing repetitive tokens or syntax echoes scores $1.0$ on $A_{\text{FT}}$ but $0.0$ on $A_{\text{FC}}$. Post-execution promotion mandates satisfying thresholds on **both** metrics simultaneously.
3. **Autoregressive Primacy:**
   All capability evaluations must be performed via free autoregressive greedy generation (`temperature=0.0`, `top_k=1`, `top_p=1.0`, `max_new_tokens=16`). Teacher-forced logit metrics are diagnostic telemetry and cannot substitute for free generation failure.

---

## 3. Metric Taxonomy & Mathematical Formulations

### 3.1 Primary Metric: First-Token Exact Match Accuracy ($A_{\text{FT}}$)
The proportion of evaluation instances where the first token emitted by the model exactly matches the ground-truth target value token ID:
$$A_{\text{FT}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{t}_{i,0} = y_{i}^*\right)$$
Where $\hat{t}_{i,0}$ is the first generated token ID and $y_i^*$ is the ground-truth target token ID.

### 3.2 Co-Primary Metric: Full Completion Match with Clean Termination ($A_{\text{FC}}$)
The proportion of instances where the model emits the target token followed immediately by `<EOS>`:
$$A_{\text{FC}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{t}_{i,0} = y_{i}^* \land \hat{t}_{i,1} = \text{eos\_id} \land \text{len}(\hat{\mathbf{t}}_i) = 2\right)$$

### 3.3 Diagnostic Metric 1: Teacher-Forced Target Token Rank ($R_{\text{tgt}}$)
Measures the rank of the correct target token in the model's pre-softmax logit distribution at the first response position under teacher forcing:
$$R_{\text{tgt}} = 1 + \sum_{v \in \mathcal{V}} \mathbb{I}\left(z_v > z_{y^*}\right)$$
- Optimal value: $R = 1$.
- Evaluated as secondary telemetry; cannot override free generation failure.

### 3.4 Diagnostic Metric 2: Prompt Sensitivity Delta ($\Delta p$)
Measures the causal influence of prompt context on output probability by swapping the target value in context ($y_A \to y_B$) while keeping the query key unchanged:
$$\Delta p = \left| P(y_A \mid \text{Prompt}_{A}) - P(y_A \mid \text{Prompt}_{B}) \right|$$
- Target threshold: $\Delta p \ge 0.20$ [PROVISIONAL].

---

## 4. Granular Error Classification Taxonomy

Every non-successful instance is automatically categorized into one of six mutually exclusive error classes:

| Error Code | Classification Name | Empirical Pattern | Interpretation |
|---|---|---|---|
| **ERR-01** | Distractor Intrusion | $\hat{t}_0 \in \{\text{Distractor Value Token IDs}\}$ | Model retrieved an irrelevant context value; attention routing failure. |
| **ERR-02** | Key Echo | $\hat{t}_0 \in \{\text{Query Key or Distractor Key IDs}\}$ | Model copied the key instead of the bound value; binding breakdown. |
| **ERR-03** | Syntax Echo | $\hat{t}_0 \in \{\text{"PAIR", "QUERY", "PAIRS", "COPY", "->" IDs}\}$ | Attention collapsed onto syntax prompt tokens; template attractor. |
| **ERR-04** | Repetition Loop | $\hat{\mathbf{t}}$ exhibits repeating n-gram cycles ($>2$ tokens) | Softmax collapse into stationary cycle states; $A_{\text{FT}}=1, A_{\text{FC}}=0$. |
| **ERR-05** | Premature Termination | $\hat{t}_0 = \text{eos\_id}$ | Immediate `<EOS>` emission; sequence termination failure. |
| **ERR-06** | Out-of-Context Prior | $\hat{t}_0 \notin \{\text{All Context Token IDs}\}$ | Unanchored language model prior emission; hallucination. |

---

## 5. Negative Controls & Baseline Benchmark Protocol

Before an evaluation run is accepted, the evaluator must execute against three negative controls:

1. **Empty-Context Negative Control:**
   Prompt: `QUERY: [KEY] ->`.
   - *Expected Behavior:* Model must not output target value at higher than unconditioned vocabulary prior probability.
2. **Wrong-Key Negative Control:**
   Prompt: `PAIR: K1=V1. QUERY: K2 ->` where $K_2$ was never defined in context.
   - *Expected Behavior:* Model must not emit $V_1$ ($ERR-01$ avoidance).
3. **Key-Shuffled Permutation Control:**
   Present prompt with all keys and values permuted across examples.
   - *Expected Behavior:* Measures background unconditioned associative baseline rate.

### 5.1 Evaluator Unit-Test Specification
The prospective evaluator script (`scripts/evaluate_cand01_benchmark.py`) must pass a dedicated unit-test suite (`tests/test_cand01_evaluator.py`) verifying that:
- It flags substring matches as non-matches when token IDs differ.
- It scores trailing tokens or repetition loops as $A_{\text{FC}} = 0.0$ while tracking $A_{\text{FT}}$.
- It correctly assigns error codes `ERR-01` through `ERR-06`.
- It aborts with an error if tokenizer vocabulary digests mismatch canonical values.

---

## 6. Two-Way Clustered Statistical Aggregation Protocol (Resolutions ADV-03 & ADV-07)

The prospective evaluator script must process all three experimental seeds (`seeds: [42, 137, 2026]`) and output:
1. **Per-Partition Metrics:** Separate evaluations for Partition C (Test Recombination) and Partition D (Test Disjoint-Token In-Context Copying).
2. **Two-Way Clustered Standard Errors:** Clustered simultaneously on `query_key_token_id` and `target_val_token_id`.
3. **Two-Way Cluster Bootstrap 95% Confidence Intervals:** Computed via $B = 2,000$ cluster resamples over key and value token IDs.
4. **Descriptive Multi-Seed Reporting:** Report individual seed scores, mean, standard deviation, SEM, and range across seeds ($n=3$). No unsupported seed-level permutation NHST claims.
5. **Mixed-Effects Logistic Regression:**
   $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_{\text{arm}} + u_{\text{key}(i)} + w_{\text{val}(j)} + \epsilon_{ij}$$

---

## 7. Evaluator Governance & Artifact Locking

- **Script Path (Prospective):** `scripts/evaluate_cand01_benchmark.py`
- **Unit Test Path (Prospective):** `tests/test_cand01_evaluator.py`
- **Benchmark Partitions (Prospective):**
  - Partition C: `cand01_benchmark_test_recomb_500.json`
  - Partition D: `cand01_benchmark_test_disjoint_500.json`
- **Locking Mandate:** The evaluator script and benchmark files must have their SHA-256 hashes recorded in the experiment completion manifest before post-training evaluation begins.
- **Execution Mandate:** This document specifies evaluation methodology only. **No evaluator execution is authorized by this document.**

