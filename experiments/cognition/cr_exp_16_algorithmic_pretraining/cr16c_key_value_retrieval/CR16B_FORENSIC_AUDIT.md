# CR-16B Forensic Audit Report

## 1. Lineage & Checkpoint Hash Audit
An independent SHA-256 calculation was conducted for all baseline, benchmark, and checkpoint artifacts referenced in CR-16B:

| Artifact | Claimed Path | Expected SHA-256 | Actual SHA-256 | Audit Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Stage B** | `experiments/batch16_21_combined/...` | `96357705a683482e...` | `96357705a683482e...` | **PASSED** |
| **Frozen CR-16 Bench** | `benchmark_data/cr16_synthetic_...` | `0b61e73d57d903f3...` | `0b61e73d57d903f3...` | **PASSED** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4...` | `9080ff51adba38d4...` | **PASSED** |
| **CR-16A Arm C** | `checkpoints/arm_c_curriculum/...` | `ac4beb4cfff904d1...` | `ac4beb4cfff904d1...` | **PASSED** |
| **CR-16A Arm D** | `checkpoints/arm_d_generalization/...` | `9d609a770c748f93...` | `9d609a770c748f93...` | **PASSED** |
| **CR-16B Benchmark** | `cr16b_copying_generalization/data/...`| `44aba606abfee0be...` | `44aba606abfee0be...` | **PASSED** |

## 2. Checkpoint Architecture & Lineage
Every checkpoint binary was inspected:
- Model configuration matches canonical ChakrMicro ($d=192, 6\text{ layers}, 6\text{ heads}, d_{\text{head}}=32, d_{\text{mlp}}=512, V=4096$).
- Pretrained weights derive strictly from canonical Stage B step 10,000.
- Tokenizer files match `data/experiments/vocab_4096/`.
- Training histories recorded in checkpoint dicts match historical run logs.

No substitution, parameter mutation, or external pretrained weight contamination was detected.

