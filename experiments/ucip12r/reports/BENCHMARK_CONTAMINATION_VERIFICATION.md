# UCIP-12R: Benchmark Contamination Verification Report

**Mission ID:** UCIP-12R — Forensic Verification  
**Audit Target:** Leakage between Stage C / SFT datasets and sealed benchmarks  
**Execution Script:** `experiments/ucip12r/scripts/verify_contamination.py`  

---

## 1. Executive Summary

Phase H performed an independent adversarial contamination scan across all 460 probes in `benchmark_r2_sealed.json` and 100 probes in `ucip11_sealed.json`.

### Audit Result: **VERIFIED ZERO LEAKAGE IN STAGE C; SFT CONTAMINATION CONFIRMED**
1. **Stage C Pretraining Foundation:** Exactly **0 prompt leaks** and **0 expected response leaks** across all 5,534 documents. Stage C is 100% clean and isolated from the sealed evaluation suites.
2. **Legacy SFT Contamination:** Independently confirmed that historical SFT files contain exact duplicates of probe `SEA_LANG_ENG_117`:
   - Prompt: *"What is a memory leak and how does it degrade system performance?"*
   - Found in: `data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl` (ID `EXP34-TECH-0018`) and related legacy SFT files.
   - **Quarantine Action Confirmed:** These legacy SFT files must remain strictly quarantined and barred from future training mixtures until sanitized.

---

## 2. Contamination Scan Matrix

| Benchmark Suite | Probe Count | Canonical SHA-256 | Overlap with Stage C | Overlap with Legacy SFT |
|---|---|---|---|---|
| **R2 Sealed Benchmark** | 460 probes | `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` | **0 leaks** | 1 prompt (`SEA_LANG_ENG_117`) |
| **UCIP-11 Sealed Suite** | 100 probes | `dce6954ab6db4aa00bfcf1405a47243d4698e9e0cd084ad63ff3e5281b678fe4` | **0 leaks** | **0 leaks** |

---

## 3. Sealed Benchmark Protection Status

- All sealed benchmark JSON files in `experiments/ucip10e_r2/` and `experiments/ucip12/` were verified by cryptographic hash.
- **Zero sealed benchmark files were modified, moved, or overwritten.**
- No full answer keys are printed into logs or reports.
