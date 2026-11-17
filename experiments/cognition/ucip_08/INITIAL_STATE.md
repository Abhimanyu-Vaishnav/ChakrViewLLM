# UCIP-08 Initial System & Preflight State

**Project:** ChakrView  
**Mission:** UCIP-08 — Independent Neural Capability Audit & Recovery  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Initial HEAD:** `fa5f3c8`  
**Date:** November 17, 2026  

---

## 1. Environment & Hardware Specifications
- **Operating System**: Windows 11 (OS: Windows)
- **CPU**: Intel Core i9-13900H / Multi-core x86_64
- **RAM**: 32 GB
- **Python**: 3.14.7
- **PyTorch**: 2.7.0.dev20250201+cpu
- **Execution Mode**: CPU-First Native Execution

---

## 2. Checkpoint & Artifact Lineage Verification

| Artifact Name | Path | Expected SHA-256 | Actual SHA-256 | Verification Status |
|---|---|---|---|---|
| **UCIP-07 Medium** | `releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt` | `554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2` | `554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2` | **100% MATCH** |
| **UCIP-07 Core** | `releases/ucip_07_candidates/chakrview_ucip07_core_model.pt` | `2657b419dc661b0be8e03bfd2ba43854c98bfa593b5ed5a971fc96ad56796c17` | `2657b419dc661b0be8e03bfd2ba43854c98bfa593b5ed5a971fc96ad56796c17` | **100% MATCH** |
| **UCIP-06 Medium** | `releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt` | `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72` | `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72` | **100% MATCH** |
| **UCIP-05 Medium** | `releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt` | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` | **100% MATCH** |
| **Canonical Stage B (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **100% MATCH** |
| **Canonical Baseline Init** | `compute_canonical_model_hash(ChakrMicro(seed=42))` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **100% MATCH** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **100% MATCH** |

---

## 3. Tokenizer Invariants
- **Vocabulary Path**: `data/experiments/vocab_4096/`
- **Vocabulary Size**: 4096 BPE tokens
- **Special Tokens**: `<unk>` (0), `<pad>` (1), `<s>` (2), `</s>` (3)
