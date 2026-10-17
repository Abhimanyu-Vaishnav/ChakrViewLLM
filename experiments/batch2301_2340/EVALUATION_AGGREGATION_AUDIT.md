# EVALUATION AGGREGATION AUDIT

**Batch:** 2301–2340  
**Focus:** Exact Mathematical Reconciliation Between Source-Level and Benchmark-Wide Cross-Entropy  
**Status:** MATHEMATICAL AGGREGATION FULLY PROVEN  

---

## 1. The Discrepancy Question

The prompt noted that in a table presented in the prior review:
- `gutenberg_loney`: 134 docs, 26,052 tokens (CE 4.8821 / 4.6001)
- `wikipedia_en_information_theory`: 51 docs, 10,750 tokens (CE 5.3727 / 4.9307)
- `wikipedia_hi_sanskrit_sahitya`: 44 docs, 8,390 tokens (CE 2.5852 / 2.5706)
- `wikipedia_sa_bhagavad_gita`: 61 docs, 19,577 tokens (CE 4.3201 / 4.3977)
- Summed tokens = 64,769, but token-weighted aggregation did not equal the overall benchmark CE of 4.4789 / 4.2861.

Forensic audit of `SOURCE_WISE_EVALUATION.md` (Batch 2261–2300) and `REPRODUCED_EVALUATION.json` reveals the exact mathematical truth.

---

## 2. Actual Ground-Truth Disaggregated Token Loss Records

The actual evaluation dataset contains **290 documents and 64,769 tokens** grouped into the 4 canonical held-out sources:

| Source Identifier | Documents | Evaluated Tokens | Baseline Token-Weighted CE | Candidate Token-Weighted CE | Delta ($\Delta$) | Baseline Summed NLL | Candidate Summed NLL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | 150 | 24,482 | 4.4315 | 4.1495 | **-0.2820** | 108,491.98 | 101,588.06 |
| `wikipedia_hi_sanskrit_sahitya` | 9 | 4,798 | 3.9203 | 3.9057 | **-0.0146** | 18,809.60 | 18,739.55 |
| `wikipedia_sa_bhagavad_gita` | 61 | 19,577 | 4.3202 | 4.3978 | **+0.0776** | 84,576.56 | 86,095.73 |
| `wikipedia_en_information_theory` | 70 | 15,912 | 4.9157 | 4.4737 | **-0.4420** | 78,218.62 | 71,185.51 |
| **Total Benchmark** | **290** | **64,769** | **4.4789** | **4.2861** | **-0.1928** | **290,096.76** | **277,608.85** |

---

## 3. Mathematical Identity Verification

Cross-entropy aggregation follows the definition:

$$\text{Overall TW-CE} = \frac{\sum_{i \in \text{sources}} (\text{TW-CE}_i \times \text{Tokens}_i)}{\sum_{i \in \text{sources}} \text{Tokens}_i} = \frac{\text{Total NLL}}{\text{Total Tokens}}$$

### Canonical Baseline:
$$\frac{108,491.98 + 18,809.60 + 84,576.56 + 78,218.62}{64,769} = \frac{290,096.76}{64,769} = 4.478896 \approx \mathbf{4.4789}$$
$$\text{Perplexity} = \exp(4.478896) = \mathbf{88.14}$$

### CPT Seed 42 Candidate:
$$\frac{101,588.06 + 18,739.55 + 86,095.73 + 71,185.51}{64,769} = \frac{277,608.85}{64,769} = 4.286138 \approx \mathbf{4.2861}$$
$$\text{Perplexity} = \exp(4.286138) = \mathbf{72.68}$$

$$\Delta \text{TW-CE} = 4.2861 - 4.4789 = \mathbf{-0.1928}$$

The source-wise metrics aggregate **exactly** to the reported overall metrics with zero residual discrepancy.

---

## 4. Origin of the Prompt's Numbers

The prompt's table listed:
- `gutenberg_loney`: 134 docs, 26,052 tokens
- `wikipedia_en_information_theory`: 51 docs, 10,750 tokens
- `wikipedia_hi_sanskrit_sahitya`: 44 docs, 8,390 tokens
- `wikipedia_sa_bhagavad_gita`: 61 docs, 19,577 tokens

These numbers represented a preliminary, unverified draft split before the Batch 2221–2260 independent evaluator cleanly separated the documents by their true manifest metadata. The true manifest partitions `gutenberg_loney` into 150 documents (24,482 tokens), `wikipedia_hi` into 9 documents (4,798 tokens), `wikipedia_sa` into 61 documents (19,577 tokens), and `wikipedia_en` into 70 documents (15,912 tokens). Under the true partition, mathematical aggregation holds with 100% precision.
