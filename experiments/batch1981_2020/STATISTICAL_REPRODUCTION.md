# CHAKRVIEW — Batches 1981–2020
## Independent Statistical Reproduction Report

### Executive Summary

In Batches 1981–2020, an independent reproduction and audit was executed on the underlying per-document predictions of the canonical checkpoint and all candidate continued-pretraining models (`cpt_balanced_seed_42`, `123`, `999`, and `cpt_balanced_seed_42_300steps`).

The statistical results were independently verified from the raw records across 88 documents and 11,110 target tokens. The two required estimands—**Token-Weighted Cross-Entropy Difference** and **Macro Document-Level Difference**—were evaluated using paired bootstrap resampling ($N=10,000$ replicates, random seed 42) at both the document level and the source-cluster level.

---

### 1. Evaluated Checkpoints & Lineage

| Model Identifier | Source Checkpoint Path | SHA-256 Checksum (Verified Bit-Exact) | Training Steps / Seed |
| :--- | :--- | :--- | :--- |
| **Canonical Baseline** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Step 10,000 |
| **CPT Seed 42** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | 150 steps / Seed 42 |
| **CPT Seed 123** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_123.pt` | `646475ee3346ea8eac2609c27968ba516b557e7c365d7c94a36771c5d6e284b8` | 150 steps / Seed 123 |
| **CPT Seed 999** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_999.pt` | `e49ce95f2d6bbe6320eebc696af30a8724facdf505f7cb0951b73b18847049ab` | 150 steps / Seed 999 |
| **CPT 300 Steps** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | 300 steps / Seed 42 |

---

### 2. Full Statistical Reproduction Results

**Baseline Performance:** Canonical Token-Weighted CE = **5.9254**, Perplexity = **374.41**, Macro Document Mean Loss = **6.3340**.

| Candidate Model | Token-Weighted CE | Token-Weighted Delta | Document 95% CI (TW) | Macro Doc Loss | Macro Doc Delta | Document 95% CI (Macro) | Formal Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPT Seed 42** | 5.7242 | **-0.2012** | `[-0.2782, -0.1321]` | 6.0252 | **-0.3088** | `[-0.3918, -0.2307]` | **Reproduced and supported** |
| **CPT Seed 123** | 5.7250 | **-0.2004** | `[-0.2821, -0.1272]` | 6.0208 | **-0.3132** | `[-0.3993, -0.2320]` | **Reproduced and supported** |
| **CPT Seed 999** | 5.7157 | **-0.2097** | `[-0.2877, -0.1400]` | 6.0160 | **-0.3180** | `[-0.4016, -0.2390]` | **Reproduced and supported** |
| **CPT 300 Steps** | 5.8420 | **-0.0834** | `[-0.1668, +0.0005]` | 6.1576 | **-0.1764** | `[-0.2537, -0.1000]` | **Reproduced and supported (Regression)** |

---

### 3. Key Findings

1. **Every Point Estimate Lies Within Its Corresponding Interval:**
   The point estimates (-0.2012 token-weighted and -0.3088 macro) fall strictly inside their respective 95% confidence intervals across all candidate models.
2. **150-Step Modest Improvement is Supported:**
   Across all three seeds, the 150-step CPT checkpoint achieves a modest, statistically significant cross-entropy reduction of approximately **-0.20 CE** (token-weighted) and **-0.31 CE** (macro document).
3. **300-Step Overfitting Regression is Confirmed:**
   Extending pretraining to 300 steps on synthetic templates causes cross-entropy to **regress by +0.1178 CE** relative to the 150-step model. In token-weighted evaluation, the 95% interval spans zero (`[-0.1668, +0.0005]`), demonstrating a complete loss of statistical significance.
4. **No Claim of General Reasoning:**
   A lower cross-entropy of -0.20 CE on language modeling demonstrates improved next-token likelihood, but does **NOT** establish mathematical reasoning, coding capability, or human-like comprehension.

*Machine-readable reproduction metrics are preserved in [REPRODUCED_METRICS.json](file:///d:/Project/ChakrView/experiments/batch1981_2020/REPRODUCED_METRICS.json).*
