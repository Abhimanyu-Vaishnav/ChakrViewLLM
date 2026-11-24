# CV-GENESIS-NR-010 Stage-C Data Audit

## 1. Executive Summary

Mission **CV-GENESIS-NR-010 Phase 1 & 2** verified the integrity, token volume, and split boundaries of the canonical Stage-C corpus used for pretraining ChakrGenesis-17M.

---

## 2. Quantitative Shard and Token Accounting

All 32 binary token shards (`np.uint16`) were audited against `data/manifests/stage_c_shards_checksums.json`:
- **Shards Verified:** **32/32 (100% SHA-256 match, 0 missing, 0 mismatches)**.
- **Total Binary Footprint:** **14.70 MB**.
- **Total Corpus Tokens:** **7,708,601 tokens**.

### Split Distribution:
- **Train (27 Shards):** **6,651,093 tokens** (86.3%)
- **Validation (2 Shards):** **465,954 tokens** (6.0%)
- **Test (3 Shards):** **591,554 tokens** (7.7%)

---

## 3. Linguistic and Domain Breakdown
- **English Prose & Technical Documents:** 2,801,696 tokens (36.4%)
- **Devanagari Hindi Texts:** 2,400,182 tokens (31.1%)
- **Code & Syntax:** 1,586,022 tokens (20.6%)
- **Reasoning & Science:** 500,122 tokens (6.5%)
- **Sanskrit & Classical Knowledge:** 407,107 tokens (5.3%)
- **Hinglish & Mathematics:** 13,472 tokens (0.2%)
