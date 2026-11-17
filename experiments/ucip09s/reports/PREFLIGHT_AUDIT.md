# UCIP-09S: Forensic Preflight Audit Report

**Mission:** UCIP-09S — Controlled Retention & Capability Isolation Study  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**HEAD Commit:** `32100ac`  
**Date:** November 17, 2026  

---

## 1. System & Execution Environment

- **CPython Version:** 3.14.7 (64-bit AMD64)
- **PyTorch Version:** 2.14.0+cpu (Native CPU execution)
- **Host OS:** Windows 11 Pro (10.0.26300-SP0)
- **Processor:** GenuineIntel Intel Core i9-13900H (32 GB RAM)
- **Execution Mode:** 100% CPU-First Indigenous Compute (Zero GPU, Zero external APIs, Zero external pretrained weights)

---

## 2. Invariant & Artifact Lineage Verification

| Artifact Name | Path | Verified SHA-256 Digest | Status |
|---|---|---|---|
| **Protected File (`session.py`)** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **100% BIT-EXACT MATCH** |
| **Primary Baseline: UCIP-09 Medium** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | **100% MATCH** |
| **Experimental: UCIP-09R Medium** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | **100% MATCH** |
| **UCIP-09 Training Dataset** | `experiments/cognition/ucip_09/data/ucip09_train.jsonl` | `2b251762e8216cfd956acdbd343e757e64bf2c33309cc24fa0f653e4ef37108d` | **100% MATCH** |
| **UCIP-09R Training Dataset** | `experiments/cognition/ucip_09r/data/ucip09r_train.jsonl` | `d53f14e2fecd4f6370081a0f12c89792d8e7db47ec3b70fc9cff50f93101bcec` | **100% MATCH** |
| **Tokenizer Config** | `data/experiments/vocab_4096/config.json` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **100% MATCH** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **100% MATCH** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **100% MATCH** |

---

## 3. Training & Architecture Parameters

- **Architecture:** `ChakrMedium`
- **Parameter Count:** 12,932,736 parameters (7 layers, width 384, 6 heads, 896 ffn)
- **Tokenizer:** 4,096 byte-level BPE (3,837 merges)
- **Decoding Configuration:** `temperature=0.0`, `repetition_penalty=1.05`, `no_repeat_ngram=3`, `max_new_tokens=48`
- **Working Tree:** Clean on `exp/cbr-01-conversational-breakthrough`
