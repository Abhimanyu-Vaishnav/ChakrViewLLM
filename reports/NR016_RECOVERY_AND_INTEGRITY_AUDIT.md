# NR-016 Phase 0 — Recovery and Artifact Integrity Audit
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `84d247c` (`CV-GENESIS-NR-015: neural learning forensics, 325-task sealed benchmark, representation analysis, curriculum experiments, and 38M architecture pilot`)  
**Upstream Synchronization:** Clean, 0 commits ahead, 0 commits behind `origin/exp/cbr-01-conversational-breakthrough`.  
**Working Tree Status:** Clean (`nothing to commit, working tree clean`).

---

## 1. Executive Summary & Audit Mandate
A comprehensive forensic audit of the repository state, Git commit history, canonical checkpoints, experimental milestones, tokenizer assets, and tensor architectures was conducted before modifying any code or running training loops.

**Key Findings:**
1. **Commit & Branch Lineage:** The local branch is at commit `84d247c`, identical to the configured remote repository at `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`.
2. **Canonical Stage-D Checkpoint Integrity:** The canonical checkpoint `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` matches the exact expected cryptographic SHA-256 signature (`2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`) and size ($68,292,655$ bytes).
3. **Resolution of the Stage-E Checkpoint Hash Discrepancy:**
   - In NR-013 report text, Stage-E was logged as `00c15185085ab8ab2d729a5fb5ed81434412eaf57e5e6b8633fc864edda9f02c`.
   - In NR-014 and NR-015 ledgers, Stage-E was logged as `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630`.
   - **Forensic Diagnosis:** The physical file on disk `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` has an exact SHA-256 of `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` ($68,292,561$ bytes). Checkpoints are excluded by `.gitignore` (`*.pt`), meaning files were never tracked or overwritten by Git commits. The `00c15185...` value was a stale hash recorded during intermediate training in NR-013 before the final training epoch completed, whereas `0ff238b6...` is the actual, physically verified, persistent file hash audited across NR-014, NR-015, and NR-016.
4. **Tokenizer & Special-Token Identity:**
   - Canonical vocabulary: `data/experiments/vocab_4096` (`vocab_size=4096`).
   - Special token mapping verified: `<BOS>=0`, `<EOS>=1`, `<PAD>=2`.
   - Base byte offset: 3.
5. **Tensor Shape & Architecture Match:**
   - Verified that `ChakrGenesis-17M` tensor shapes match config exactly:
     - `embedding.weight`: `torch.Size([4096, 384])`
     - 10 layers, each with:
       - `attn.q_proj.weight`: `torch.Size([384, 384])` (12 query heads $\times 32$)
       - `attn.k_proj.weight`: `torch.Size([96, 384])` (3 KV heads $\times 32$, GQA 4:1)
       - `attn.v_proj.weight`: `torch.Size([96, 384])` (3 KV heads $\times 32$, GQA 4:1)
       - `attn.out_proj.weight`: `torch.Size([384, 384])`
       - `ffn.gate_proj.weight`: `torch.Size([1024, 384])` (SwiGLU hidden dim 1024)
       - `ffn.up_proj.weight`: `torch.Size([1024, 384])`
       - `ffn.down_proj.weight`: `torch.Size([384, 1024])`
       - Pre-RMSNorm `attn_norm` & `ffn_norm`: `torch.Size([384])`

---

## 2. Cryptographic Checkpoint Inventory

| Checkpoint Identifier | File Path | File Size (Bytes) | SHA-256 Checksum | Verified Lineage / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Pretrained (5.18M)** | `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` | 62,204,051 | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` | Canonical 5.18M base |
| **NR-005 SFT Best (5.18M)** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 62,203,157 | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Conversational foundation |
| **NR-010 Pretrain Best (17M)** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 68,292,851 | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | 17M Pretrained foundation |
| **NR-010 SFT Best (17M)** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | 68,292,297 | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` | 17M SFT foundation |
| **ChakrGenesis-17M Stage-D Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 68,292,655 | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | **PRIMARY CANONICAL CHECKPOINT** |
| **ChakrGenesis-17M Stage-E Best** | `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` | 68,292,561 | `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` | Archived Experimental (NR-013) |
| **ChakrGenesis-17M Stage-F Best** | `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` | 68,292,561 | `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` | Archived Experimental (NR-014) |

---

## 3. Epistemic Assessment & Integrity Confirmation
All seven canonical and historical checkpoints are intact, uncorrupted, and physically present in the workspace. No Git LFS divergence or uncommitted state exists. The repository is in an ideal posture to execute the controlled Phase 1 forensic reproduction.
