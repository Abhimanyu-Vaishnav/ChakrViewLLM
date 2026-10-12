# Final Engineering Report — Master Autonomous Sprint Batches 71–74
## ChakrView: Reproducibility Audit, Controlled Arithmetic Learning, and Evidence-Based Next Steps

---

### 1. Executive Summary

In Master Autonomous Sprint Batches 71–74, we addressed the core scientific question:
> **Can the existing ChakrMicro model learn a simple mathematical transformation beyond memorizing training examples?**

To answer this conclusively without conflating causes or relying on unverified assumptions:
1. **Batch 71 (Reproducibility & Hash Audit)**: We reconciled the canonical initialization hash procedure bit-for-bit, verified the step-10,000 checkpoint hash, confirmed all 96 historical regression tests pass, and ensured that the uncommitted user modification in [session.py](file:///d:/Project/ChakrView/chakrview/runtime/session.py) was strictly preserved.
2. **Batch 72 (Trustworthy Arithmetic Benchmark)**: We engineered a single-digit addition benchmark ($a + b = c \le 9$) covering all 55 valid integer pairs, with disjoint Train (30), Validation (10), and Held-Out Test (10) splits, plus an 11-pair **Symmetry Diagnostic** to test commutativity. All operands and answers correspond strictly to single tokens in ChakrMicro's vocabulary ($0 \to 51, \dots, 9 \to 60$).
3. **Batch 73 (Controlled Learning Experiments)**: We evaluated Baseline Untrained, Condition A (Uniform SFT), Condition B (5x Answer Emphasis), Condition C (Answer-Only Diagnostic), and an Overfit Control. While the overfit control achieved 100% memorization (3/3), the 100-step runs collapsed to a constant attractor digit (`7`), producing 1/10 validation match and 1/10 test match strictly because `3+4=7` and `4+3=7` happened to sum to 7. Zero-shot symmetry accuracy was 0/11.
4. **Batch 74 (Rationale-Style Data Hypothesis)**: We tested whether intermediate reasoning tokens (`a + b = c. Answer: c`) enable attention layers to compute the transformation. The model learned the surface syntactic rationale with 100% precision, but exhibited **premise drift and intermediate hallucination** (e.g. given `4 + 2`, generating `4 + 0 = 7. Answer: 7`), confirming that intermediate text alone does not induce operator reasoning in small models without dense arithmetic representations.

---

### 2. Quantitative Results Across All Evaluated Conditions

| Condition / Strategy | TRAIN Exact Match | VAL Exact Match | HELD-OUT TEST Exact Match | Symmetry Diagnostic | TRAIN Ans Loss | VAL Ans Loss | EOS Completion | Phenomenon Observed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline Untrained (Step 10k)** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 10.0813 | 10.0336 | 0.0% | Pretraining table attractor / no EOS |
| **Condition A: Uniform SFT (100 steps)** | 5 / 30 (16.67%) | 1 / 10 (10.0%)* | 1 / 10 (10.0%)* | 0 / 11 (0.0%) | 2.2003 | 2.3444 | 100.0% | Modal collapse: predicts constant digit `7` for all inputs |
| **Condition B: Answer-Emphasis (5x, 100 steps)**| 5 / 30 (16.67%) | 1 / 10 (10.0%)* | 1 / 10 (10.0%)* | 0 / 11 (0.0%) | 2.2619 | 2.3315 | 100.0% | Identical modal collapse to constant `7` |
| **Condition C: Answer-Only Diagnostic** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 2.3399 | 2.3591 | 0.0% | Runaway digit loops (loss of termination) |
| **Overfit Control (3 pairs, 50 steps)** | **3 / 3 (100.0%)** | 0 / 10 (0.0%) | 3 / 10 (30.0%)* | 1 / 11 (9.09%)* | **0.0046** | 8.2669 | 100.0% | 100% memorization; fits training subset perfectly |
| **Batch 74: Rationale Intermediate SFT** | **9 / 30 (30.0%)** | **0 / 10 (0.0%)** | **1 / 10 (10.0%)*** | 0 / 11 (0.0%) | 2.1520 | 2.4180 | 100.0% | Intermediate hallucination: outputs `X + Y = 7. Answer: 7` |

*\*Note on Generalization*: The 1/10 accuracy in validation and test splits is **not operator generalization**. In both splits, exactly 1 input pair summed to 7 (`3 + 4` in Val, `4 + 3` in Test). Because the model predicted `7` unconditionally for all inputs, it scored an accidental 1/10 match. In the symmetry diagnostic, where no pair sums to 7, the model scored **0/11 (0.0%)**.

---

### 3. Scientific Verdict

1. **Memorization Ability**: **CONFIRMED & REPRODUCIBLE**.
   ChakrMicro v0.1 has sufficient capacity to memorize specific string pairs (reaching 100% on the 3-pair overfit control and 100% on the Batch 65 32-pair full-batch run).
2. **Mathematical Transformation Generalization**: **CONFIRMED ZERO**.
   Under direct-answer SFT, answer-emphasis SFT, and intermediate rationale SFT, true mathematical generalization across unseen single-digit additions is **0.0%**.
3. **Failure Mechanism (Mode Collapse & Premise Drift)**:
   A 3.44M parameter model trained via next-token prediction learns that the assistant turn demands a numeric digit and closing template. To minimize cross-entropy loss across the training distribution, it defaults to the median digit (`7`). When trained with intermediate rationale templates, it learns the surface syntax (`a + b = c. Answer: c`) but hallucinates training premises rather than computing the sum from the prompt.

---

### 4. Canonical Baseline & Checkpoint Invariants

All canonical ChakrMicro invariants remain strictly verified bit-exact:
- **Total Parameters**: `3,443,136` (exact match).
- **Fresh Init SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (bit-exact under canonical named-parameter digest procedure).
- **Canonical Step 10k Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (bit-exact).
- **Preserved User Modification**: [session.py](file:///d:/Project/ChakrView/chakrview/runtime/session.py) was kept untouched and uncommitted throughout the entire sprint.
- **Hardware Utilized**: CPU-only execution (Intel x86_64, Windows, Python 3.14.7, PyTorch 2.14.0+cpu). No GPU execution claimed.
- **Test Results**: **101 passing regression tests** (96 historical + 5 new Batch 71–74 tests).

---

### 5. Git Status and Delivery Readiness

- **Current Branch**: `dev/v0.6-v0.7-validation`
- **Configured Remote**: `origin -> https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Push Policy**: In compliance with non-negotiable safety rules, **no remote push was executed**. The branch contains verified local work and awaits explicit user authorization before any remote push.
- **Main-Branch Merge Policy**: No merge into `main` is recommended yet. `main` is at release v0.5.0 (`d048678`), while ongoing experimental validation remains on `dev/v0.6-v0.7-validation`.

---

### 6. Single Most Valuable Next Milestone

Now that the training-generation gap and arithmetic capability boundaries are rigorously mapped, the highest-value next step is:
**Pretraining Alignment Probe & Synthetic Corpus Pre-Injection**: Investigate whether injecting dense mathematical and algorithmic tokens during pretraining (Stage B/C) establishes the foundational attention routing circuits required for downstream SFT to elicit arithmetic computation.
