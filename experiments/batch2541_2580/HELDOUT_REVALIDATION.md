# HELD-OUT METRICS INDEPENDENT REVALIDATION REPORT
**Batch:** 2541–2580  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation` @ `3b30f86`)  
**Primary Source:** `experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl` (290 test records)  
**Manifests Cross-Referenced:** `SPLIT_V2_MANIFEST.json`, `CORPUS_V2_CLEANED_MANIFEST.json`  

---

### 1. Executive Summary

We performed an independent, ground-up recomputation of evaluation metrics across all 290 held-out test documents. No values were hardcoded or copied from prior markdown summaries. All records were checked for:
1. Exact bidirectional set equality against `SPLIT_V2_MANIFEST.json` ($290 \iff 290$).
2. Strict source identifier fidelity against `CORPUS_V2_CLEANED_MANIFEST.json`.
3. Non-negative, finite float constraints on $NLL$ and cross-entropy loss values.
4. Internal mathematical consistency: $\text{NLL} = \text{Loss} \times \text{Target Tokens} \pm 10^{-4}$.

---

### 2. Global Evaluation Results

| Metric | Canonical Baseline (Step 10,000) | Candidate (Seed 42 CPT) | Delta ($\Delta$) | Rel. Change (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Test Documents** | 290 | 290 | — | — |
| **Total Target Tokens** | 64,769 | 64,769 | — | — |
| **Cross-Entropy (CE)** | 4.4789 | 4.2861 | **-0.1928** | -4.30% |
| **Perplexity (PPL)** | 88.14 | 72.68 | **-15.46** | -17.54% |
| **Candidate Document Wins** | — | 222 / 290 | — | **76.55%** |

---

### 3. Per-Domain Breakdown

| Source Identifier | Language | Docs | Target Tokens | Baseline CE | Candidate CE | Delta CE ($\Delta$) | Win Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gutenberg_loney_coordinate_geometry` | English | 150 | 24,482 | 4.4315 | 4.1495 | **-0.2820** | 92.00% (138/150) |
| `wikipedia_en_information_theory` | English | 70 | 15,912 | 4.9157 | 4.4737 | **-0.4420** | 90.00% (63/70) |
| `wikipedia_hi_sanskrit_sahitya` | Hindi | 9 | 4,798 | 3.9203 | 3.9057 | **-0.0147** | 66.67% (6/9) |
| `wikipedia_sa_bhagavad_gita` | Sanskrit | 61 | 19,577 | 4.3202 | 4.3978 | **+0.0776** | **24.59%** (15/61) |

---

### 4. Key Analytical Findings

1. **Sanskrit Degradation in Candidate Model:**
   - On Classical Sanskrit (`wikipedia_sa_bhagavad_gita`), the Seed 42 CPT candidate regresses by **+0.0776 CE** (a $+1.80\%$ degradation in loss), winning on only 15 of 61 documents (24.59%).
   - This empirically confirms the hypothesis that general CPT without explicit Sanskrit representation degraded performance on Classical Sanskrit text.
2. **Hindi Control Stability:**
   - Hindi text on Sanskrit literature (`wikipedia_hi_sanskrit_sahitya`) improves slightly (**-0.0147 CE**, win rate 66.67%), proving that the Devanagari script representation itself did not broadly catastrophically collapse, but Sanskrit grammar/vocabulary was specifically neglected.
3. **Discrepancy Audit Against Batch 2501–2540:**
   - All newly recomputed values are in bit-exact alignment with Batch 2501 primary derivations (`global_delta_ce = -0.192791`, `sa_delta = +0.077592`, `hi_delta = -0.014661`).
