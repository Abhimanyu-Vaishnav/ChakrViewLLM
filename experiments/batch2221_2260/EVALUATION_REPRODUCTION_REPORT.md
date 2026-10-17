# EVALUATION REPRODUCTION REPORT

**Batch:** 2221–2260  
**Date:** October 2026  
**Auditor:** ChakrView Senior Systems & ML Research Engineer  
**Evaluation Mode:** Fresh, Independent Execution (`experiments/batch2221_2260/run_independent_evaluation.py`)  
**Corpus Manifest:** `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` (SHA-256: `e3db033c...`)  
**Tokenizer:** Canonical `vocab_4096` (`release/chakrview-v0.5.0/data/experiments/vocab_4096/`)  
**Evaluated Device:** CPU (`torch.inference_mode()`, single-pass deterministic inference)  
**Output Artifact:** `experiments/batch2221_2260/REPRODUCED_EVALUATION.json`  

---

## 1. Executive Summary

This report documents a **fresh, independent reproduction** of the held-out test evaluation across the canonical baseline, historical CPT candidates, and all three Batch 2141–2180 CPT candidates.

Rather than copying historical summaries, the reproduction script freshly loaded each model checkpoint from disk, verified its configuration, tokenized all 290 documents from the cleaned corpus manifest, and evaluated cross-entropy across all **64,769 tokens**.

### Key Confirmed Findings
1. **Replication of CPT Gains:** All three CPT seeds independently reproduced their reported compression gains:
   - **Canonical Baseline:** `4.4789` TW-CE (PPL: 88.14)
   - **CPT Seed 42 (Step 125):** `4.2861` TW-CE (PPL: 72.68, $\Delta = -0.1928$)
   - **CPT Seed 1337 (Step 50):** `4.3046` TW-CE (PPL: 74.04, $\Delta = -0.1743$)
   - **CPT Seed 2026 (Step 50):** `4.3067` TW-CE (PPL: 74.20, $\Delta = -0.1722$)
2. **Document-Level Win Rate & Bootstrap CI:**
   - Seed 42 improved **222 out of 290 test documents (76.55%)**.
   - Mean paired document loss difference: $\bar{D} = -0.2362$.
   - 95% bootstrap confidence interval ($B=10,000$): `[-0.2630, -0.2095]`.
   - Seed 1337 improved 231 documents (79.66%), $\bar{D} = -0.2087$, 95% CI `[-0.2301, -0.1874]`.
   - Seed 2026 improved 245 documents (84.48%), $\bar{D} = -0.2031$, 95% CI `[-0.2220, -0.1845]`.
3. **Historical Regression Confirmed:** Historical CPT 150-step and 300-step checkpoints from Batches 1861–1940 both exhibited severe regression (+0.0704 and +0.1783 CE increases; win rates of only 18.97% and 3.79%), confirming that earlier CPT protocols overfit.

---

## 2. Recomputed Primary Benchmark Table

| Model Identity | Evaluated Checkpoint File | Evaluated Docs | Evaluated Tokens | Token-Weighted CE | Perplexity (PPL) | Macro-Source CE | Mean Paired Doc $\Delta$ | 95% Paired Doc Bootstrap CI | Document Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | `training_checkpoint_step_10000.pt` | 290 | 64,769 | **4.4789** | **88.14** | **4.3969** | 0.0000 | Baseline | Baseline |
| **Historical CPT 150** | `cpt_balanced_seed_42.pt` | 290 | 64,769 | 4.5493 | 94.57 | 4.4524 | +0.0854 | [+0.0746, +0.0960] | 18.97% (55/290) |
| **Historical CPT 300** | `cpt_balanced_seed_42_300steps.pt` | 290 | 64,769 | 4.6572 | 105.34 | 4.5474 | +0.2099 | [+0.1922, +0.2271] | 3.79% (11/290) |
| **CPT Seed 42 (Step 125)**| `cpt_best_checkpoint.pt` | 290 | 64,769 | **4.2861** | **72.68** | **4.2317** | **-0.2362** | **[-0.2630, -0.2095]**| **76.55% (222/290)**|
| **CPT Seed 1337 (Step 50)**| `cpt_best_checkpoint.pt` | 290 | 64,769 | **4.3046** | **74.04** | **4.2458** | **-0.2087** | **[-0.2301, -0.1874]**| **79.66% (231/290)**|
| **CPT Seed 2026 (Step 50)**| `cpt_best_checkpoint.pt` | 290 | 64,769 | **4.3067** | **74.20** | **4.2478** | **-0.2031** | **[-0.2220, -0.1845]**| **84.48% (245/290)**|

---

## 3. Disaggregated Source-Family Performance

| Source Family ID | Source Domain | Doc Count | Token Count | Canonical Baseline | Hist 150 | Hist 300 | Seed 42 | Seed 1337 | Seed 2026 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | Mathematics | 150 | 24,482 | 4.4315 | 4.5562 | 4.7086 | **4.1495 (-0.2820)** | **4.1772 (-0.2543)** | **4.1845 (-0.2470)** |
| `wikipedia_en_information_theory` | Tech / Math | 70 | 15,912 | 4.9157 | 4.9959 | 5.1463 | **4.4737 (-0.4420)** | **4.5541 (-0.3616)** | **4.5928 (-0.3229)** |
| `wikipedia_hi_sanskrit_sahitya` | Hindi Essays | 9 | 4,798 | 3.9203 | 3.9270 | 3.9713 | **3.9057 (-0.0146)** | **3.8890 (-0.0313)** | **3.8833 (-0.0370)** |
| `wikipedia_sa_bhagavad_gita` | Sanskrit Classical | 61 | 19,577 | **4.3202** | 4.3303 | 4.3634 | 4.3978 (+0.0776) | 4.3631 (+0.0429) | 4.3307 (+0.0105) |

---

## 4. Verification Conclusion

The fresh independent evaluation confirms:
1. The reported language model compression improvements in Batch 2141–2180 are **100% reproducible and mathematically verified**.
2. All evaluated token counts (64,769) and document counts (290) match the manifest exactly.
3. The paired document bootstrap intervals are strictly negative and exclude zero across all three CPT seeds.
