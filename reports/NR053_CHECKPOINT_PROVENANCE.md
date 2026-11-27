# NR-053: Checkpoint Provenance Audit Report

**Mission:** CHAKRVIEW NR-053 — Generalization, Data Leakage & Overfitting Forensic Validation  
**Date:** November 27, 2026  
**Auditor:** ChakrGenesis Independent Forensic Review Team  
**Artifact Path:** [checkpoint_provenance.json](file:///D:/Project/ChakrView/experiments/forensics_nr053/checkpoint_provenance.json)

---

## 1. Executive Summary

This audit establishes the lineage, initialization history, dataset dependencies, and control integrity across all 5 primary candidates in the ChakrView research series.

### Key Forensic Findings
1. **Uncontrolled Comparison: RC3 vs Scratch Candidates:**
   - **RC3** was initialized from `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` (`2c21632e...`), which underwent extensive historical multi-stage pretraining and curriculum tuning.
   - In contrast, NR-051 and NR-052 were trained from random initialization (`seed=42`).
   - Comparing RC3 against scratch candidates is an uncontrolled comparison because RC3 carries the inductive bias of historical pretraining.
2. **Controlled Comparisons:**
   - **NR-051 17M vs NR-051 38M:** Strictly controlled head-to-head comparison on the uncurated pretraining dataset.
   - **NR-052 17M vs NR-052 38M:** Strictly controlled head-to-head comparison on the clean continuous prose Corpus V4.
   - **NR-051 vs NR-052:** Controlled comparison isolating the impact of pretraining data cleaning (eliminating template multipliers and boilerplate).

---

## 2. Checkpoint Provenance Matrix

| Candidate ID | Actual Checkpoint Hash (SHA-256) | Initializer | Architecture | Pretrain Corpus | SFT Dataset | Controlled Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RC3 Baseline** | `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c` | Stage-D Best (`2c21632e`) | 17M (8L, 384D) | Legacy Stage-D | Corpus V3 | **HISTORICAL BASELINE** |
| **NR-051 17M SFT**| `9f63e097108f4c3ef2fdc882826d570845a101d02f036bc1c055ad66183600b4` | NR051 17M Base (`8d871ca0`) | 17M (8L, 384D) | NR-051 Mix (1.05M tok) | Corpus V3 | Controlled vs NR-051 38M |
| **NR-051 38M SFT**| `a0639a3fba90352c47b151fb70e60209a1889cd3265b80c3cadcb751e3c5b31f` | NR051 38M Base (`22306c9a`) | 38M (12L, 512D) | NR-051 Mix (1.05M tok) | Corpus V3 | Controlled vs NR-051 17M |
| **NR-052 17M SFT**| `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb` | NR052 17M Base (`58781710`) | 17M (8L, 384D) | Corpus V4 (624k tok) | Corpus V3 | Strictly Controlled vs NR-052 38M |
| **NR-052 38M SFT**| `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed` | NR052 38M Base (`7fb29b35`) | 38M (12L, 512D) | Corpus V4 (624k tok) | Corpus V3 | Strictly Controlled vs NR-052 17M |
