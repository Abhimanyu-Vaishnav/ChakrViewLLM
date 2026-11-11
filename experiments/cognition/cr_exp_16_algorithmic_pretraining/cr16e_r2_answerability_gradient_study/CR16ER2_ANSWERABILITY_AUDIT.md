# CR-16E-R2 Phase B: Correct Answerability Labels Audit

### 1. Executive Answerability Audit Findings

In previous reports (CR-16E and CR-16E-R), false abstention on the sealed CR-16C benchmark was calculated assuming all 1,500 probes were answerable ($N=1,500$ denominator).

**Audited Discovery:**
- CR-16C contains **10 distinct categories**, one of which is `CAT9: missing_key_abstention` ($100$ probes).
- In `CAT9`, the queried key is absent from the context, and the expected target is explicitly `[UNKNOWN]`.
- Treating all 1,500 probes as answerable falsely classified correct abstentions on CAT9 as "false abstentions", distorting the false abstention rate and undercounting answerable selectivity.

---

### 2. Comprehensive Benchmark Label Verification Matrix

Every individual probe across all six evaluation benchmarks was audited and labeled as one of:
`ANSWERABLE`, `UNANSWERABLE`, `AMBIGUOUS`, or `INVALID_PROBE`.

#### Table B1: Validated Answerability Breakdown Across Suites

| Benchmark Suite | Total Probes | ANSWERABLE | UNANSWERABLE | AMBIGUOUS | INVALID_PROBE | Audit Notes |
|---|---|---|---|---|---|---|
| **CR-16C Sealed Retrieval** | **1,500** | **1,400 (93.33%)** | **100 (6.67%)** | **0** | **0** | CAT1–8 & CAT10 are Answerable; CAT9 ($N=100$) is strictly Unanswerable. |
| ├── CAT1 Direct Familiar | 200 | 200 | 0 | 0 | 0 | Target fact exists in context at Pos 0. |
| ├── CAT2 Reassigned Values | 200 | 200 | 0 | 0 | 0 | Update value exists in context. |
| ├── CAT3 Counterfactual Updates | 150 | 150 | 0 | 0 | 0 | Update value exists in context. |
| ├── CAT4 Multiple Stored Facts | 150 | 150 | 0 | 0 | 0 | 4 facts present; target key bound. |
| ├── CAT5 Unseen Key-Val Combos | 150 | 150 | 0 | 0 | 0 | Target key and value both present. |
| ├── CAT6 Unseen Keys Fam Vals | 150 | 150 | 0 | 0 | 0 | Target key and value both present. |
| ├── CAT7 Fam Keys Unseen Vals | 150 | 150 | 0 | 0 | 0 | Target key and value both present. |
| ├── CAT8 Distractor Heavy | 150 | 150 | 0 | 0 | 0 | Distractor syntax present; key bound. |
| ├── CAT9 Missing Key Abstention | 100 | 0 | **100** | 0 | 0 | Queried key absent; target `[UNKNOWN]`. |
| └── CAT10 Unseen Templates | 100 | 100 | 0 | 0 | 0 | Novel syntax; target value present. |
| **B1 Position Invariance** | **600** | **600 (100.0%)** | **0** | **0** | **0** | Target key bound in slots 0, 1, 2, or 3. |
| **B2 Counterfactual Pairs** | **300 (150 pairs)** | **300 (100.0%)** | **0** | **0** | **0** | Target value present in probe A and B. |
| **B2 Permutation Pairs** | **300 (150 pairs)** | **300 (100.0%)** | **0** | **0** | **0** | Target value present in probe A and B. |
| **B3 Multi-Token Spans** | **500** | **500 (100.0%)** | **0** | **0** | **0** | Target entity present in context. |
| **B4 Calibrated Abstention** | **300** | **0** | **300 (100.0%)** | **0** | **0** | Queried key absent from context. |
| **CR-16 Copying Benchmark** | **100** | **100 (100.0%)** | **0** | **0** | **0** | Target string is exact input copy. |

- Full machine-readable per-probe labels saved to: `answerability_labels.json`.
- Full category-wise summary counts saved to: `answerability_audit_summary.json`.
- Zero ambiguous probes and zero invalid probes were detected.

