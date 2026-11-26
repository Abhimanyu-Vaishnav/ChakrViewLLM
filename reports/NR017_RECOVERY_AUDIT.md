# NR-017 Phase 0 — Recovery and Integrity Audit
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `ca95e99` (`CV-GENESIS-NR-016: controlled capacity comparison (17M vs 38M), 325-benchmark recovery, scientific generator, and gate decision`)  
**Upstream Synchronization:** Clean, 0 commits ahead, 0 commits behind `origin/exp/cbr-01-conversational-breakthrough`.  
**Working Tree Status:** Clean (`nothing to commit, working tree clean`).  
**Isolated Experiment Directory:** `experiments/genesis_stage_g_017/` (Created)  
**Isolated Data Directory:** `data/stage_g_017/` (Created)

---

## 1. Executive Summary & Checkpoint Verification
Before initiating any new task generation, state-trace formulation, or training experiments, the complete repository state, Git commit history, canonical checkpoints, experimental milestones, and tensor architectures were audited.

**Verified Artifact Signatures:**
1. **Primary Canonical Foundation Checkpoint:**
   - Path: `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`
   - Size: $68,292,655$ bytes
   - SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`
   - Status: **VERIFIED CANONICAL FOUNDATION**
2. **Historical Experimental Milestones:**
   - Stage-E Best: `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` ($68,292,561$ bytes, SHA-256: `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630`)
   - Stage-F Best: `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` ($68,292,561$ bytes, SHA-256: `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8`)
   - Pretrain 17M: `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` ($68,292,851$ bytes, SHA-256: `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0`)
   - Canonical 5.18M Pretrain: `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` ($62,204,051$ bytes, SHA-256: `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c`)
3. **Canonical Tokenizer Identity:**
   - Vocabulary: `data/experiments/vocab_4096` (`vocab_size=4096`, base byte offset 3).
   - Special tokens: `<BOS>=0`, `<EOS>=1`, `<PAD>=2`.
4. **Primary Model Architecture:**
   - `ChakrGenesis-17M`: $17,063,808$ parameters, 10 layers, $d_{model}=384$, 12 Q-heads, 3 KV-heads (GQA 4:1), $d_{ff}=1024$ (SwiGLU), RoPE, Pre-RMSNorm.

---

## 2. Epistemic Mandate for CV-GENESIS-NR-017
- All experiments will be strictly conducted in the newly created `experiments/genesis_stage_g_017/` and `data/stage_g_017/` directories.
- Zero modifications to canonical checkpoints or historical artifacts.
- The mission will directly address the central question: **Does explicit, verifiable state-trace supervision enable algorithmic reasoning and out-of-distribution generalization in ChakrGenesis-17M compared with direct-answer training?**
