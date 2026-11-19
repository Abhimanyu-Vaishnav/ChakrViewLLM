# UCIP-13: Preflight & Dataset Verification Report

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** 2026-11-19  

---

## 1. Executive Summary

Phase 1 of UCIP-13 independently verified all preflight constraints, canonical artifact hashes, and Stage C dataset partitions prior to beginning from-scratch autoregressive pretraining.

---

## 2. Cryptographic Preflight Sign-Off

All critical hashes were audited:

| Component | Filesystem Target | Computed SHA-256 | Expected SHA-256 | Status |
|---|---|---|---|---|
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51...` | **MATCH (Frozen, $\Delta W = 0$)** |
| **Canonical Baseline** | `ChakrMicro v0.1 (seed 42)` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c...` | **MATCH** |
| **Stage C Manifest** | `data/manifests/stage_c_manifest.json` | `6c744118f0fc93c24cf425f9613644306c8b010c33d01f0421c157bcadb49296` | `6c744118...` | **MATCH** |
| **R2 Sealed Suite** | `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` | `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` | `ef940768...` | **MATCH (Frozen)** |
| **UCIP-11 Sealed Suite**| `experiments/ucip11/benchmarks/sealed/ucip11_sealed.json` | `dce6954ab6db4aa00bfcf1405a47243d4698e9e0cd084ad63ff3e5281b678fe4` | `dce6954a...` | **MATCH (Frozen)** |

---

## 3. Dataset Shard Verification

The Stage C shards in `data/tokenized/stage_c/` were verified against partition boundaries:
- **Train Partition:** 27 shards (`shard_00000.bin` to `shard_00026.bin`), totaling 6,651,093 tokens.
- **Validation Partition:** 2 shards (`shard_00000.bin` and `shard_00001.bin`), totaling 465,954 tokens.
- **Held-Out Test Partition:** 3 shards (`shard_00000.bin` to `shard_00002.bin`), totaling 591,554 tokens (strictly quarantined).
