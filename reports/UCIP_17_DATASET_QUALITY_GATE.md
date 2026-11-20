# UCIP-17 Dataset Quality Gate Report

**Mission ID:** UCIP-17  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Dataset Path:** `data/sft/ucip17/`  
**Auditor:** Antigravity  
**Audit Timestamp:** 2026-11-20T18:32:55+05:30  
**Quality Gate Status:** PASSED (Ready for Controlled Pilot Proposal)  

---

## 1. Quantitative Verification & Partition Invariants

All 15,000 reconstructed records in `data/sft/ucip17/` were audited and verified under automated pytest test suites:

| Partition File | Record Count | Line Count | SHA-256 Checksum | Audit Status |
|---|---:|---:|---|:---:|
| `sft_train_12000.jsonl` | 12,000 | 12,000 | `0ad8840efb43d8c2d9a7c12d785683b6791a906e18f06376974286ed0638eb80` | **MATCH** |
| `sft_validation_1500.jsonl` | 1,500 | 1,500 | `972affab6e583475bab151d4f88606adfb3e6846bc4157c36dc4f9a6ab7c21f2` | **MATCH** |
| `sft_dev_1500.jsonl` | 1,500 | 1,500 | `cd84f301f638405516e0b76195862d1938491bc66129d58c7449b5d0bd108370` | **MATCH** |
| `sft_master_15000.jsonl` | 15,000 | 15,000 | `6b6a614f6009350171f596faf5029e152ef6b57357e085e9d38ad41b9eac5d39` | **MATCH** |

---

## 2. Category & Language Balance

- **Category Balance (18/18):** Exact allocation matched for all categories (`INST` 1,200, `PYTHON` 1,200, `HIN` 1,200 down to `SAN` 450).
- **Language Balance (4/4):** English (8,250 = 55.0%), Hindi (4,500 = 30.0%), Hinglish (1,500 = 10.0%), Sanskrit (750 = 5.0%).
- **Zero Partition Leakage:** $Train \cap Val = \emptyset$, $Train \cap Dev = \emptyset$, $Val \cap Dev = \emptyset$.
- **Zero Sealed Benchmark Contamination:** Audited against all 7,617 historical benchmark probes. Contamination count: **0** ($0.0\%$).
- **Context Length Bounding:** $100\%$ of sequences fit within the 512-token context ceiling.

---

## 3. Stratified Manual Review of 360 Records

A stratified sample of 360 records (20 sampled across each of the 18 categories) was inspected:
1. **Mathematical Accuracy (`ARITH`):** 20/20 verified for exact arithmetic operations.
2. **Syntactic Validity (`PYTHON`):** 20/20 parsed through Python AST `compile()` cleanly.
3. **Script Appropriateness (`HIN`/`SAN`):** 20/20 contain authentic Devanagari Unicode.
4. **Diversity Metric Comparison:**
   - UCIP-14R Training: 1,851 unique responses (15.4% unique ratio).
   - UCIP-17 Training: **3,141 unique responses** (significant expansion of prompt/response combinations).
