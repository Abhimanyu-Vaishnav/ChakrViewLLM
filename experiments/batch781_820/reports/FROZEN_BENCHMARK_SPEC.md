# Frozen Benchmark Specification: Batches 781–820
## Compositional Test Matrix, Scenario Families & Integrity Audit

**Date:** 2026-10-15  
**Milestone:** Batches 781–820  
**Repository:** `ChakrView`  
**Location:** `experiments/batch781_820/reports/FROZEN_BENCHMARK_SPEC.md`

---

### 1. Benchmark Overview and Frozen Checksum

The held-out evaluation suite for Batches 781–820 comprises **$N=70$** frozen, compositional task-planning queries.

- **File Path:** `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
- **Frozen File SHA-256:** `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a`
- **Audit Verification:** Strict $0.0\%$ prompt and target leakage against both training ($N=380$) and validation ($N=50$) splits.

---

### 2. Compositional Matrix & Length Distribution

| Scenario Family | Description | Step Count | Items ($N$) |
| :--- | :--- | :---: | :---: |
| **`math_2step`** | Disjoint two-step arithmetic (operands in $[600, 614]$) | 2 | 15 |
| **`math_3step`** | Disjoint three-step arithmetic (operands in $[700, 714]$) with $\$0, \$1$ dependencies | 3 | 15 |
| **`math_4step`** | Disjoint four-step arithmetic pipeline (operands in $[800, 811]$) with $\$0, \$1, \$2$ dependencies | 4 | 12 |
| **`text_chain`** | Disjoint text transformation chains with unique held-out technical phrases | 2 | 10 |
| **`clarification`** | Incomplete invoice settlement queries lacking required fields (e.g. VAT, customs duty) | 1 | 6 |
| **`unsupported`** | Adversarial exploit and root authorization requests requiring safe refusal | 1 | 6 |
| **`hindi_multistep`** | Multi-step arithmetic queries formulated in Hindi | 3 | 6 |
| **Total** | | **1–4** | **70** |

---

### 3. Frozen Benchmark Integrity Constraints
1. **Zero Contamination:** Test examples are strictly forbidden from training loss calculations or hyperparameter selection.
2. **Explicit Denominators:** All metrics will use the full $N=70$ denominator unless reporting a dedicated family subset (e.g., $N=6$ for unsupported actions).
3. **Immutability:** The frozen SHA-256 (`cabf7daa...`) will be verified before and after all evaluation runs.
