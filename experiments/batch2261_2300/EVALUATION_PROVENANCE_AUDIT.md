# EVALUATION PROVENANCE AUDIT

**Batch:** 2261–2300  
**Audit Date:** October 2026  
**Auditor:** ChakrView Senior Research and Release Auditor  
**Machine-Readable Audit Record:** `experiments/batch2261_2300/EVALUATION_AUDIT.json`  

---

## 1. Checkpoint Provenance & Cryptographic Identity

All six evaluated model configurations were traced directly to their on-disk checkpoint files. Cryptographic SHA-256 hashes and file sizes confirm that each configuration is a distinct, verifiable artifact rather than an alias:

| Model Configuration | Relative Checkpoint Path | File Size (Bytes) | SHA-256 Checksum | Provenance / Best Step |
| :--- | :--- | :---: | :--- | :--- |
| **Canonical Baseline** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 42,212,787 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Canonical pretraining (10,000 steps) |
| **Historical CPT 150** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | 13,794,913 | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | Batch 1861 historical balanced CPT |
| **Historical CPT 300** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | 13,795,727 | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | Batch 1901 historical extended CPT |
| **CPT Seed 42 Candidate** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | 41,402,267 | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | Batch 2141 CPT (Best validation: Step 125) |
| **CPT Seed 1337 Candidate** | `experiments/batch2141_2180/cpt/seed_1337/cpt_best_checkpoint.pt` | 41,402,267 | `05c021c37b61f6e6869833a27d46380c3eaa23c062b6aa4341a3c5a755e0f28b` | Batch 2141 CPT (Best validation: Step 50) |
| **CPT Seed 2026 Candidate** | `experiments/batch2141_2180/cpt/seed_2026/cpt_best_checkpoint.pt` | 41,402,203 | `c47e826d37279ef6c639cb802dae112adb8d253be9ad38d9b3ebc266af4291f9` | Batch 2141 CPT (Best validation: Step 50) |

---

## 2. Evaluation Population & Benchmark Verification

- **Manifest File:** `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` (SHA-256: `e3db033c...`)
- **Document Count:** Exactly **290 documents** assigned to `split == "test"`.
- **Token Count:** Exactly **64,769 tokens** evaluated across all 290 documents.
- **Source Families:**
  1. `gutenberg_loney_coordinate_geometry`: 150 documents (24,482 tokens)
  2. `wikipedia_en_information_theory`: 70 documents (15,912 tokens)
  3. `wikipedia_sa_bhagavad_gita`: 61 documents (19,577 tokens)
  4. `wikipedia_hi_sanskrit_sahitya`: 9 documents (4,798 tokens)

---

## 3. Fresh Reproduction vs Historical Metrics

Fresh independent evaluation was executed via `experiments/batch2221_2260/run_independent_evaluation.py`. The comparison between fresh measurements and historical claims demonstrates zero deviation:

| Model Configuration | Historical TW-CE | Fresh TW-CE | Historical PPL | Fresh PPL | Fresh Document Win Rate | Fresh Paired Bootstrap 95% CI | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | 4.4789 | **4.4789** | 88.14 | **88.14** | Baseline | Baseline | **PASS** |
| **Historical CPT 150** | 4.5493 | **4.5493** | 94.57 | **94.57** | 18.97% (55/290) | [+0.0746, +0.0960] | **PASS** |
| **Historical CPT 300** | 4.6572 | **4.6572** | 105.34 | **105.34** | 3.79% (11/290) | [+0.1922, +0.2271] | **PASS** |
| **CPT Seed 42 Candidate** | 4.2861 | **4.2861** | 72.68 | **72.68** | 76.55% (222/290) | **[-0.2630, -0.2095]** | **PASS** |
| **CPT Seed 1337 Candidate** | 4.3046 | **4.3046** | 74.04 | **74.04** | 79.66% (231/290) | **[-0.2301, -0.1874]** | **PASS** |
| **CPT Seed 2026 Candidate** | 4.3067 | **4.3067** | 74.20 | **74.20** | 84.48% (245/290) | **[-0.2220, -0.1845]** | **PASS** |

### Statistical Discipline Audit:
- **Token-Weighted Aggregate Loss Delta:** $\Delta_{\text{token}} = -0.1928$.
- **Mean Paired Document Difference:** $\bar{D} = -0.2362$.
- **95% Bootstrap Interval:** Explicitly confirmed to describe the **unweighted paired document delta** ($\bar{D}$), not the token-weighted aggregate delta.
- **Statistical Significance:** Paired $t$-test confirms $t = -17.51, p < 10^{-45}$.

**Evaluation Provenance Status:** `PASS` (100% Bit-Identical Independent Reproduction).
