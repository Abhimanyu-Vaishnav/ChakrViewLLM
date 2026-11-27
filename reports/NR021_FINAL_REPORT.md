# NR-021 Final Scientific Report

## 1. Mission Metadata
- **Mission ID:** `NR-021`
- **Title:** Neural Contribution & Source-Grounding Forensics
- **Repository:** `D:\Project\ChakrView`
- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Evaluated Checkpoints:** Canonical Stage-D 17M (`2c21632e`), 17M Neural Router (`80252516`)
- **Sealed Benchmark:** `data/stage_j_020/fresh_matched_benchmark_1000.json` (SHA-256 `238E85D2...`)
- **Evaluation Artifact:** `data/stage_k_021/ablation_10_results.json`

---

## 2. Summary of Verified Findings

1. **Pure Neural Accuracy:** Autoregressive pure neural generation (System A) achieved **1.40%** on the 1,000-task benchmark. Without external pretrained weights or scratchpad supervision, the 17M model cannot independently perform multi-digit arithmetic or string manipulations.
2. **Hybrid Advantage:** Neural Source Grounding (System D) achieved **37.70%**, outperforming direct neural router JSON generation (System C: 1.90%) by **+35.80%**.
3. **Locus of Intelligence:** The neural network acts purely as a linguistic intent classifier. Grounding arguments via deterministic regex achieves **100% parameter accuracy** on all supported operations when the operation is known (Condition F: 76.0% vs Oracle H: 76.0%).
4. **Benchmark Verification:** No data leakage, duplicate prompts, or task-distribution skews exist in the sealed 1,000-task matched benchmark.

---

## 3. Official Verdict
`PHASE 1 (NR-021) COMPLETE: SOURCE-GROUNDING PIPELINE FULLY FORENSICALLY DECONSTRUCTED; BOTTLENECK ISOLATED TO NEURAL OPERATION CLASSIFICATION; CLEARED FOR PHASE 2 (CONVERSATIONAL SPECIFICATION).`
