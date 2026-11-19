# UCIP-12R: Stage C Binary Shard Integrity Report

**Mission ID:** UCIP-12R — Forensic Verification  
**Audit Target:** All binary shards in `data/tokenized/stage_c/`  
**Execution Script:** `experiments/ucip12r/scripts/verify_stage_c_shards.py`  
**Manifest Artifact:** `experiments/ucip12r/manifests/shard_verification.json`  

---

## 1. Executive Summary

Phase B conducted an exhaustive, byte-level forensic verification of every single binary shard in Stage C. Every shard was individually inspected, unpacked as little-endian uint16 integers, tested for vocabulary boundary compliance ($0 \le \text{token\_id} < 4096$), checked for exact byte alignment, and evaluated for cryptographic hash uniqueness.

### Audit Result: **100% VERIFIED**
- **Total Shards:** Exactly **32 shards** (27 train, 2 validation, 3 test).
- **Total Token Count:** Exactly **7,708,601 uint16 tokens** (15,417,202 bytes).
- **Exact Reconciliation:** Exactly **7,703,067 content tokens** + **5,534 document EOS tokens** ($1$).
- **Byte Alignment:** $100\%$ of files have even byte counts ($\text{size} = N \times 2$).
- **Invalid Tokens:** Exactly **0** tokens $\ge 4096$.
- **Shard Uniqueness:** Exactly **32 unique SHA-256 hashes** (0 duplicate files).

---

## 2. Partition Breakdown & Shard Statistics

| Partition | Shard Count | Total Bytes | Total Tokens | Min ID | Max ID | EOS ($1$) Count | Usable Autoregressive Tokens |
|---|---|---|---|---|---|---|---|
| **Train** | 27 shards | 13,302,186 | 6,651,093 | 1 | 4094 | 4,755 | **6,651,093** |
| **Validation** | 2 shards | 931,908 | 465,954 | 1 | 4094 | 378 | **465,954** |
| **Test** | 3 shards | 1,183,108 | 591,554 | 1 | 4094 | 401 | **591,554** |
| **TOTAL** | **32 shards** | **15,417,202** | **7,708,601** | **1** | **4094** | **5,534** | **7,708,601** |

---

## 3. Shard Size & Representation Details

- Standard shard sizing: Shards `shard_00000.bin` through `shard_00025.bin` in `train/` contain exactly **500,000 bytes (250,000 tokens)** each.
- The terminal shard `train/shard_00026.bin` contains the remainder of 302,186 bytes (151,093 tokens).
- Validation contains 1 standard shard (250,000 tokens) and 1 remainder shard (215,954 tokens).
- Test contains 2 standard shards (250,000 tokens each) and 1 remainder shard (91,554 tokens).
- Binary format: Little-endian 16-bit unsigned integers (`<H`), exactly matching PyTorch `torch.int64` / `torch.int16` tensors after loading via `numpy.fromfile(dtype=np.uint16)`.

---

## 4. Special Token & Vocabulary Audit

- **PAD ($2$):** 0 padding tokens present in raw shards; documents are packed sequentially with EOS boundaries.
- **EOS ($1$):** Exactly **5,534 occurrences** across the entire 32 shards, corresponding 1:1 to the 5,534 documents in `stage_c_manifest.json`.
- **Vocabulary Bounds:** Minimum token ID is $1$ (EOS); maximum token ID is $4094$. Zero tokens exceed the 4096 vocabulary ceiling.
- **Conclusion:** Stage C binary shards represent a mathematically flawless, byte-aligned, lossless autoregressive training corpus.
