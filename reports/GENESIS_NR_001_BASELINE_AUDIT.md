# GENESIS-NR-001: Forensic Baseline Audit Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Execution Environment:** `.venv_gpu` (PyTorch 2.5.1+cu124, CUDA 12.6, NVIDIA RTX A1000 6GB Laptop GPU)  

---

## 1. Executive Forensic Summary

This investigation isolated the exact root cause of the severe conversational failure where the application generated repetitive attractor outputs such as `Gou I hoon! I hoon! I hoon!` and `Gou I ast se shuru I hoon!`.

### Key Forensic Findings:
1. **Application Checkpoint Divergence:**
   - In `releases/project_genesis_candidate/web_app.py`, line 26 was mistakenly importing `ChakrViewUltimateAgent` from `chakrview.interface.ultimate.agent_controller` instead of `ChakrGenesisAgent` from `chakrview.interface.genesis.agent_controller`.
   - As a direct consequence, the web application was loading the outdated `chakr_ultra_best.pt` model (trained on only 545 synthetic SFT examples with zero prior language pretraining) rather than the newly pretrained and fine-tuned `genesis_sft_best.pt` model.
2. **Attractor Mechanism in `chakr_ultra_best.pt`:**
   - Because `chakr_ultra_best.pt` was trained on 545 SFT examples without autoregressive language pretraining, its output distribution suffered severe logit collapse.
   - When given greetings or conversational inputs, the model predicted token 74 (`'G'`), token 120 (`'ou'`), token 35 (`' '`), token 76 (`'I'`), followed by repetition loops of token 441 (`' hoon!'`).

---

## 2. Artifact Registry & SHA-256 Hashes

| Artifact Description | Path | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Active SFT Checkpoint** | `experiments/project_genesis/conversation_training/genesis_sft_best.pt` | 20,733,097 bytes | `ebde2fb8e156ad1f74baa5245c3e4fd9fb35f3caa4e381b8261bbf3d0b2c28b4` |
| **Candidate SFT Checkpoint** | `releases/project_genesis_candidate/genesis_sft_best.pt` | 20,733,097 bytes | `ebde2fb8e156ad1f74baa5245c3e4fd9fb35f3caa4e381b8261bbf3d0b2c28b4` |
| **Pretraining Best Checkpoint** | `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` | 62,204,051 bytes | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` |
| **Pretraining Final Checkpoint**| `experiments/project_genesis/pretraining/genesis_pretrain_final.pt` | 20,733,533 bytes | `11c6f633492443cac11025bbc8f092d68393f4727fa7c1b18903c2e15dab21fa` |
| **Legacy Checkpoint (Ultrabrain)**| `releases/ultimate_brain_candidate/chakr_ultra_best.pt` | 20,736,231 bytes | `df33f5d6febeee396b26c043477140e69ca38346df8cf85b2e987c2fb240409a` |
| **Tokenizer Vocabulary** | `experiments/tokenizer/candidates/v4096/vocab.json` | 112,969 bytes | `fbde6847bb4e678edd2590908f472c0883d74c6abcb92c9b626269ea18a28adb` |
| **Tokenizer Merges** | `experiments/tokenizer/candidates/v4096/merges.json` | 76,310 bytes | `259f537b23e1186e719501f3994639cc537a2626316f32ffcec98d85a9274a89` |
| **Genesis Architecture** | `chakrview/brain/genesis/chakr_genesis.py` | 8,623 bytes | `fd32372f6745b76f9910529ab010077dd371d54d45958be8ce75262f184546f3` |
| **Genesis Agent Controller** | `chakrview/interface/genesis/agent_controller.py` | 7,286 bytes | `82d1a34b249762ff6d6e676f1b40b192e269fe29a46b15abf5f3b3154d1b113e` |
| **Web Application Script** | `releases/project_genesis_candidate/web_app.py` | 8,810 bytes | `25c0f1d15d5a42f3a1b5bcfe6c12eb6e922a3aa9c6734f74f0fe9d36407ff813` |
