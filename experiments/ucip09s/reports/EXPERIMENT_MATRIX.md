# Stage E — Controlled Retention Experiment Matrix

**Date:** 2026-11-17  
**Mission:** UCIP-09S Controlled Retention & Capability Isolation Study  
**Objective:** Formulate and validate controlled experimental conditions to test whether balanced replay prevents catastrophic forgetting while acquiring targeted capabilities.

---

## 1. Experimental Conditions Design

We define 4 distinct conditions to systematically isolate the variables responsible for degradation:

| Condition ID | Name | Parent Checkpoint | Dataset Mixture Composition | Size (Train) | Target / Retention Ratio |
|---|---|---|---|:---:|:---:|
| **Condition A** | Baseline Benchmark (Parent) | UCIP-08 Medium | UCIP-09 Balanced Dataset | 3,240 | Parent Reference |
| **Condition B** | Narrow Fine-Tuning (Replication) | UCIP-09 Medium | UCIP-09R Skewed Dataset (88.5% Math, 5.6% Copy, 5.9% Mixed) | 885 | 94.1% Targeted / 5.9% Anchor |
| **Condition C** | **Balanced Replay (Proposed Solution)** | UCIP-09 Medium | Interleaved: 70% UCIP-09 Core Conversational/Multilingual + 30% Curated Targeted Arithmetic & Copying | 1,200 | **30.0% Targeted / 70.0% Anchor Replay** |
| **Condition D** | Targeted Only with High Regularization | UCIP-09 Medium | Targeted arithmetic & copying only, but with high weight decay & reduced LR ($1.0 \times 10^{-5}$) | 885 | 100% Targeted / 0% Anchor Replay |

---

## 2. Hypothesis & Expected Behavioral Outcomes

1. **Hypothesis 1 (Condition B Failure Confirmation):**
   - Fine-tuning without retention anchors causes conversational accuracy to drop significantly on the 20-probe owner suite (as verified in Stage B: 55% $\to$ 30%).
2. **Hypothesis 2 (Condition C Retention & Generalization Success):**
   - Incorporating a 70% anchor replay buffer from UCIP-09 preserves English conversation (100%), Hindi Devanagari generation ($\ge 66\%$), and identity coherence, while enabling the model to learn arithmetic structure without numeric collapse (`21` / `22` bias).
   - Target goal: $\ge 60.0\%$ on the 20-probe owner suite and $\ge 15.0\%$ on the sealed 500-probe benchmark.
3. **Hypothesis 3 (Condition D Inadequacy):**
   - Regularization alone (lower LR, higher weight decay) without data replay cannot prevent representational drift when the training objective is 100% concentrated on a narrow domain.

---

## 3. Dataset Construction Specification for Condition C (`ucip09s_balanced_replay.jsonl`)

The dataset for Condition C is constructed as follows:
- **Anchor Retention Replay (70% = 840 samples):**
  - English Conversation & Identity: 240 samples
  - Devanagari Hindi: 240 samples
  - Hinglish & Conversational Systems: 200 samples
  - Instruction Following: 160 samples
- **Targeted Capabilities (30% = 360 samples):**
  - Canonical Arithmetic (Diverse operands, single & multi-digit, addition/subtraction): 240 samples
  - Exact Copying (Exact sentence verbatim repetition): 120 samples
- **Total Dataset Size:** 1,200 samples.
- **Validation Dataset (`ucip09s_balanced_val.jsonl`):** 120 balanced samples (70% anchor / 30% targeted).

---

## 4. Evaluation Protocol

All candidate models will be evaluated using identical, deterministic pipelines:
1. **Frozen 20-Probe Owner Suite:**
   - 4 English Conversation
   - 5 Arithmetic
   - 3 Exact Copying
   - 3 Devanagari Hindi
   - 2 Hinglish
   - 3 Instruction Following
2. **500-Probe Sealed Neural Benchmark:**
   - Full evaluation across all 500 frozen probes via `scripts/run_m3_evaluations.py`.
3. **Regression Safety Checks:**
   - Runtime architecture compatibility test (`tests/test_architecture_loader.py`).
   - Protected file integrity check (`session.py`).
