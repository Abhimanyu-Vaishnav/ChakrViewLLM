# Batch 88 — Incremental Supervision vs Direct Learning Results

## 1. Experimental Overview

Batch 88 evaluated whether single-step incremental state supervision enables ChakrMicro v0.1 (3,443,136 parameters) to learn single-digit addition.

Both Condition A and Condition B were trained for 100 steps from the immutable canonical step-10k checkpoint on matched budgets of 124,289 tokens:
- **Condition A (Direct Arithmetic)**: Checkpoint SHA-256: `e97b6a46469c0b211dce484fd0f8ee1b489130ca5dc52f872c171121e3f0b739`.
- **Condition B (Incremental Supervision)**: Checkpoint SHA-256: `f0bb3b4ffb1af0f382efe4406875bb418b6a980936e2e9295ea808d5cdc390e6`.

---

## 2. Quantitative Results Across All Evaluated Splits

### 2.1 Direct-Answer Evaluation

| Condition | Train Exact Match | Val Exact Match | Held-Out Test Match | EOS Rate | Predominant Answer |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0.0% | Wikitable loops |
| **Condition A (Direct)** | 7 / 64 (10.9%) | 0 / 18 (0.0%) | 2 / 18 (11.1%)* | 100.0% | **`10` (100% modal collapse)** |
| **Condition B (Incremental)** | 4 / 64 (6.2%) | 2 / 18 (11.1%) | **0 / 18 (0.0%)** | 100.0% | **`13` (100% modal collapse)** |

*\*Note on Condition A*: 100% of outputs collapsed to `10`. The 2/18 test match was an artifact of test pairs whose sum was 10 (`1+9=10`, `9+1=10`).

---

### 2.2 Autonomous Multi-Step Incremental Chain Evaluation

In this evaluation, the model had to execute each transition step autoregressively, feeding its own generated state into the next step prompt without oracle intervention:

| Condition | Train Chain Success | Val Chain Success | Held-Out Test Chain Success | Test Step Transition Accuracy | Output Collapse |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | 58 / 64 (90.6%)* | 15 / 18 (83.3%)* | 17 / 18 (94.4%)* | 0.0% (Unparsed) | Fallback loop artifact |
| **Condition A (Direct)** | 7 / 64 (10.9%) | 0 / 18 (0.0%) | 2 / 18 (11.1%) | 0 / 83 (0.0%) | `10` (100% modal) |
| **Condition B (Incremental)** | 4 / 64 (6.2%) | 2 / 18 (11.1%) | **0 / 18 (0.0%)** | **0 / 83 (0.0%)** | **`13` (100% modal)** |

*\*Note on Baseline*: The untuned baseline produced unparsed Wikitable markup; an internal fallback treated None as a static pass-through, inflating chain score without valid numerical steps.

---

## 3. Mechanistic Diagnosis of Condition B (Incremental Supervision)

1. **Immediate Modal Attractor Formation**:
   - Despite supervising individual incremental transitions ($k \to k + 1$), the model collapsed its transition predictions onto `13`.
   - When prompted `Add step 1: 0 + 8 from 0`, the model emitted `11` or `13`.
   - When prompted `Add step 2: 0 + 8 from 11`, the model emitted `13`.
2. **Failure of Input-Conditioned Succession**:
   - The model failed to learn the successor function $S(n) = n + 1$.
   - Held-Out Test exact chain success was **0 / 18 (0.0%)**, and valid transition accuracy on unseen inputs was **0.0%**.
3. **Scientific Takeaway**:
   - In small autoregressive language models (3.44M parameters), next-token cross-entropy on simple number strings causes the loss landscape to quickly find a central attractor token (here, `13`) that minimizes cross-entropy across all step targets, rather than learning the arithmetic successor operation.
   - Cold reload verified this deterministically.
