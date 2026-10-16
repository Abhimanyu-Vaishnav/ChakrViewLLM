# CHAKRVIEW — Batches 1901–1940
## Independent Evaluation Audit Report

### 1. Executive Summary

This mission performed an independent, evidence-based audit of the language-modeling claims reported in Batches 1861–1900.

The previous report claimed that continued pretraining (CPT) reduced held-out cross-entropy from **5.7817** to **4.0067** (after 150 steps) and **3.0989** (after 300 steps) on a 12,479-token corpus.

Our audit established the following verified facts:
1. **Mathematical Metrics Replicated Exactly:** Using the exact original frozen corpus and checkpoints, cross-entropy reproduced bit-for-bit:
   - Canonical Baseline: **5.7817** (Perplexity 324.30, Top-1 10.22%)
   - 150-Step CPT (Mean of 3 seeds): **4.0067** (Perplexity 54.98, Top-1 30.77%)
   - 300-Step CPT (Condition D): **3.0989** (Perplexity 22.17, Top-1 46.36%)
2. **Evaluation Implementation is Free of Target Leakage:** Attention masking is strictly causal, loss is computed as a strict token-weighted sum across non-padding tokens, and target alignment correctly pairs position $t$ with label $t+1$.
3. **Severe Evaluation Corpus Contamination Discovered:**
   - **Prose Domain:** 100% of the held-out prose test documents (8/8) were identical duplicate boilerplate placeholder strings (`Domain prose indigenous system text specifications verifying architecture standards for ChakrView milestone batches.`).
   - **Remaining Domains:** Train and test chunks were drawn from identical underlying synthetic files (`stage_b_systems.txt`, `stage_b_derivations.txt`, etc.), sharing identical sentence templates repeated hundreds of times.
4. **Generalization Collapses on Independent Benchmark:**
   When evaluated on a newly constructed, strictly source-disjoint benchmark (88 documents, 11,110 tokens) drawing only from untouched raw files:
   - Canonical Baseline: **5.9254** (PPL 374.41)
   - 150-Step CPT (Seed 42): **5.7242** (PPL 306.17) — modest improvement of **0.2012** CE ($p < 0.05$).
   - 300-Step CPT (Seed 42): **5.8420** (PPL 344.46) — **regresses by 0.1178 CE vs 150-step**, exhibiting severe overfitting to training templates.
5. **Condition D Checkpoint Was Unpersisted:** The 300-step checkpoint was not written to disk in Batch 1861; it was re-trained, audited, and serialized in Batch 1901 (`cpt_balanced_seed_42_300steps.pt`).

---

### 2. Checkpoint and Metric Reproduction Audit

#### Checkpoint Verification
- **Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-exact).
- **Adapted Reference Checkpoint:** `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`
  - SHA256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` (Verified bit-exact).
- **Batch 1861 CPT Checkpoints (150 steps):**
  - Seed 42: `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` (Verified bit-exact).
  - Seed 123: `646475ee3346ea8eac2609c27968ba516b557e7c365d7c94a36771c5d6e284b8` (Verified bit-exact).
  - Seed 999: `e49ce95f2d6bbe6320eebc696af30a8724facdf505f7cb0951b73b18847049ab` (Verified bit-exact).
- **Batch 1901 Persisted 300-Step Checkpoint:**
  - `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt`
  - SHA256: `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628`

#### Evaluation Implementation Audit
- **Target Alignment:** Evaluated sequence chunks use `chunk[:-1]` as inputs and `chunk[1:]` as targets. Loss is computed on the logits of the final hidden representations.
- **Aggregation:** Per-token cross-entropy is summed (`reduction="sum"`) and normalized by total non-padding evaluated tokens across all documents.
- **Top-k Accuracy:** Top-1 and Top-5 accuracy strictly compare next-token vocabulary argmax against ground-truth targets.

---

### 3. Comprehensive Evaluation Comparison

| Evaluation Benchmark | Model Checkpoint | Cross-Entropy | Perplexity | Top-1 Accuracy | Delta vs Canonical |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch 1861 Test Set** | Canonical Baseline | 5.7817 | 324.30 | 10.22% | Baseline |
| (48 docs, 12,479 tokens) | CPT Seed 42 (150s) | 4.0426 | 56.97 | 30.30% | -1.7391 |
| *(Contaminated Corpus)* | CPT Seed 123 (150s) | 3.9906 | 54.09 | 31.00% | -1.7911 |
| | CPT Seed 999 (150s) | 3.9869 | 53.89 | 31.00% | -1.7948 |
| | **CPT Mean (150s)** | **4.0067** | **54.98** | **30.77%** | **-1.7750** |
| | CPT Seed 42 (300s) | **3.0989** | **22.17** | **46.36%** | **-2.6828** |
| --- | --- | --- | --- | --- | --- |
| **Independent Benchmark** | Canonical Baseline | 5.9254 | 374.41 | 10.26% | Baseline |
| (88 docs, 11,110 tokens) | CPT Seed 42 (150s) | 5.7242 | 306.17 | 9.62% | -0.2012 |
| *(Source-Disjoint Corpus)* | CPT Seed 123 (150s) | 5.7250 | 306.44 | 9.59% | -0.2004 |
| | CPT Seed 999 (150s) | 5.7157 | 303.61 | 9.67% | -0.2097 |
| | **CPT Mean (150s)** | **5.7216** | **305.41** | **9.63%** | **-0.2038** |
| | CPT Seed 42 (300s) | **5.8420** | **344.46** | **9.69%** | **-0.0834** |

---

### 4. Paired Bootstrap Analysis on Independent Benchmark

Using 1,000 paired bootstrap resamples on the 88 documents of the source-disjoint benchmark:
- **CPT 150-Step vs Canonical:**
  - Mean Document Loss Difference: **-0.3088** (95% CI: `[-0.3915, -0.2279]`).
  - Statistically significant but modest improvement across unshared prose, technical, and multilingual text.
- **CPT 300-Step vs Canonical:**
  - Mean Document Loss Difference: **-0.1763** (95% CI: `[-0.2491, -0.1013]`).
  - **Regression vs 150-Step:** The 300-step checkpoint loses over **43%** of its independent gain, proving that scaling from 150 to 300 steps on the synthetic corpus overfits to training templates at the expense of external generalization.

---

### 5. Transfer Diagnostic Audit

Across the 13 diagnostic tasks:
- Canonical: **0 / 13 (0.0%)**
- 150-Step CPT: **1 / 13 (7.7%)** (Only `self.items = ` -> `[`)
- 300-Step CPT: **1 / 13 (7.7%)** (Only `self.items = ` -> `[`)
- Arithmetic, multi-step math, and proposition recall completely failed across all checkpoints.
