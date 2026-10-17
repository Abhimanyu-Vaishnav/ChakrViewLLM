# ChakrView Evaluation Protocol and Results — Batches 2141–2180
**Milestone:** Batches 2141–2180  
**Phase:** Workstream C — Evaluation that Measures Real Change  
**Date:** October 2026  
**Auditor:** Lead ML Research & Release Auditor Agent  

---

## 1. Evaluation Architecture & Blinding Policy

In strict conformance with `CPT_PREREGISTRATION.json`, the final evaluation benchmark remained completely blinded throughout training and checkpoint selection.

### Evaluated Model Suite:
1. **Canonical Baseline:** Step 10,000 pristine pretraining baseline (`training_checkpoint_step_10000.pt`, SHA-256: `96357705...`)
2. **Historical CPT 150-Step:** Checkpoint from synthetic pretraining (`cpt_balanced_seed_42.pt`, SHA-256: `85753315...`)
3. **Historical CPT 300-Step:** Checkpoint from synthetic pretraining (`cpt_balanced_seed_42_300steps.pt`, SHA-256: `2f775151...`)
4. **CPT Seed 42 Candidate:** Step 125 selected checkpoint (SHA-256: `d3268fc4...`)
5. **CPT Seed 1337 Candidate:** Step 50 selected checkpoint (SHA-256: `05c021c3...`)
6. **CPT Seed 2026 Candidate:** Step 50 selected checkpoint (SHA-256: `c47e826d...`)

---

## 2. Frozen Held-Out Test Split Results

Evaluated across all 290 documents and 65,059 target tokens in the cleaned test split (`CORPUS_V2_CLEANED_MANIFEST.json`):

| Evaluation Dimension | Canonical Baseline | Historical 150-Step | Historical 300-Step | CPT Seed 42 (step 125) | CPT Seed 1337 (step 50) | CPT Seed 2026 (step 50) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Token-Weighted CE Loss** | **4.4789** | 4.5493 | 4.6572 | **4.2861** | **4.3046** | **4.3067** |
| **Token Perplexity (PPL)** | **88.14** | 94.57 | 105.34 | **72.68** | **74.04** | **74.20** |
| **Macro-Source CE Loss** | **4.3969** | 4.4524 | 4.5474 | **4.2317** | **4.2458** | **4.2478** |
| **Macro-Source Perplexity**| **81.20** | 85.83 | 94.39 | **68.83** | **69.82** | **69.95** |
| **Net Test Delta vs Baseline** | - | +0.0704 | +0.1783 | **-0.1928 (Improved)** | **-0.1743 (Improved)** | **-0.1722 (Improved)** |

### Disaggregated Source-Family Breakdown:
| Source Family (Domain) | Tokens | Canonical Baseline | Hist 150 | Hist 300 | Seed 42 | Seed 1337 | Seed 2026 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gutenberg_loney_coordinate_geometry` (Math) | 24,632 | 4.4315 | 4.5562 | 4.7086 | **4.1495 (-0.2820)** | **4.1772 (-0.2543)** | **4.1845 (-0.2470)** |
| `wikipedia_en_information_theory` (Math/Tech)| 15,982 | 4.9157 | 4.9959 | 5.1463 | **4.4737 (-0.4420)** | **4.5541 (-0.3616)** | **4.5928 (-0.3229)** |
| `wikipedia_hi_sanskrit_sahitya` (Hindi Lit)  | 4,807  | 3.9203 | 3.9270 | 3.9713 | **3.9057 (-0.0146)** | **3.8890 (-0.0313)** | **3.8833 (-0.0370)** |
| `wikipedia_sa_bhagavad_gita` (Sanskrit)      | 19,638 | **4.3202** | 4.3303 | 4.3634 | 4.3978 (+0.0776) | 4.3631 (+0.0429) | 4.3307 (+0.0105) |

---

## 3. Statistical Significance & Paired Uncertainty Analysis

Comparing CPT Seed 42 versus Canonical Baseline on the 290 held-out test documents:
- **Documents Improved:** 222 / 290 (**76.6%**)
- **Documents Regressed:** 68 / 290 (23.4%)
- **Mean Paired Loss Difference:** $\bar{D} = -0.2362$
- **95% Document Bootstrap Confidence Interval ($B=10,000$):** `[-0.2622, -0.2099]`
- **Analytical Paired $t$-test:** $t = -17.51, p < 10^{-45}$.

**Key Empirical Finding:**
The test loss improvement of the new CPT candidates is mathematically significant and stable across all three random seeds. Loney geometry improved by $-0.28$ CE and Information Theory improved by $-0.44$ CE. Classical Sanskrit exhibited slight regression ($+0.077$ CE on Seed 42, $+0.010$ CE on Seed 2026), reflecting the trade-off of English/Hindi technical pretraining exposure without Sanskrit in the training split.

---

## 4. Capability Probes: Separating LM Loss from Functional Reasoning

A lower cross-entropy does not automatically mean a model has acquired general reasoning or tool adherence. We evaluated the models against 4 targeted capability probes under deterministic greedy decoding ($T=0$):

| Evaluation Suite | Sample Count | Scoring Rule | Canonical Baseline | Hist 150 | Hist 300 | Seed 42 | Seed 1337 | Seed 2026 |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arithmetic Probes** | 20 | Exact string match of target numeral | 1 / 20 (5.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Python Syntax Completion**| 15 | Syntactic validity via `ast.parse` | 5 / 15 (33.3%) | 3 / 15 (20.0%) | 7 / 15 (46.7%) | 2 / 15 (13.3%) | 6 / 15 (40.0%) | 2 / 15 (13.3%) |
| **Factual Cloze Retrieval** | 10 | Target entity match in generated span | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 1 / 10 (10.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) |
| **Generation Token Diversity**| 3 prompts | Unique tokens / 32 tokens ratio | 0.500 | 0.198 | 0.448 | 0.344 | 0.292 | 0.219 |

### Honest Capability Findings:
1. **Arithmetic:** At 3.4M parameters, raw greedy neural decoding has near-zero zero-shot arithmetic accuracy. Solving arithmetic requires ChakrView's neuro-symbolic calculator tool rather than bare weights.
2. **Code Syntax:** Syntax validity remains between 13% and 40%, reflecting subword completion variance. Seed 1337 achieved 40% syntax validity.
3. **Factual Cloze:** Factual recall across short clozes is near zero, proving that 200 continued pretraining steps does not turn a 3.4M model into an encyclopedic retrieval engine.
4. **Conclusion:** CPT successfully improved multi-domain language representation, statistical compression, and held-out predictability, but **did not endow the raw neural weights with autonomous multi-step reasoning**.
