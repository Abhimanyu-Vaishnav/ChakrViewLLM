# UCIP-12: Data Splitting, Isolation & Benchmark Contamination Audit

**Mission:** UCIP-12 — Large-Scale Corpus Foundation & Learning Readiness  
**Target Suites:**  
- `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` (460 probes)  
- `experiments/ucip11/benchmarks/sealed/ucip11_sealed.json` (100 probes)  

---

## 1. Threat Model & Audit Mandate

Benchmark contamination (either explicit prompt memorization or implicit semantic leakage) creates the illusion of model capability while destroying out-of-distribution generalization.
In UCIP-10E-R1, forensic analysis revealed that 81 prompts in the sealed suite were verbatim copies of training data.
For UCIP-12, an exhaustive cross-partition leakage scan was mandated across all training data partitions.

---

## 2. Methodology

1. **Probe Extraction:**
   - 460 probes extracted from R2 Sealed Suite (`benchmark_r2_sealed.json`).
   - 100 probes extracted from UCIP-11 Sealed Suite (`ucip11_sealed.json`).
   - All prompt strings (>15 characters) and unique expected answers extracted and normalized (case-folded, whitespace stripped).
2. **Corpus Traversal:**
   - Evaluated against 110 raw, processed, and SFT files across the entire `data/` and `experiments/` hierarchy.
   - Evaluated against all 5,534 Stage C documents in `data/manifests/stage_c_manifest.json`.

---

## 3. Findings

### A. Stage C Pretraining Foundation (7.7M Tokens)
- **Exact Leaks:** **0** (Zero).
- **Near-Duplicate / Substring Overlaps:** **0**.
- **Conclusion:** The entire 7.7M token pretraining corpus is completely free of sealed benchmark contamination. Models pretrained on Stage C will encounter genuinely unseen language tasks during evaluation.

### B. Legacy SFT Datasets
- 5 exact string matches were identified in historical instruction tuning files:
  - Probe: `SEA_LANG_ENG_117` (*"What is a memory leak and how does it degrade system performance over time?"*)
  - Contaminated Files:
    - `data/sft/exp34_instruction_mixture_train.jsonl`
    - `data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl`
    - `data/sft/exp34_candidate_v3/exp34_instruction_mixture_train_v3.jsonl`
    - `data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl`
    - `data/sft/cbr_sft_02/splits/cbr_sft_02_train_540.jsonl`
- **Quarantine Action:** These historical SFT files are marked contaminated and must **NOT** be included in future clean training mixtures without filtering out contaminated probe IDs.

---

## 4. Frozen Sealed Benchmark Establishment for UCIP-12

To ensure absolute reproducibility and isolation for all subsequent missions:
1. Copied canonical clean R2 Sealed suite to `experiments/ucip12/benchmarks/sealed/benchmark_r2_sealed.json`.
2. Verified exact SHA-256 hash match:
   - SHA-256: `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef`
3. Created `experiments/ucip12/benchmarks/benchmark_manifest.json` locking the sealed hash.
