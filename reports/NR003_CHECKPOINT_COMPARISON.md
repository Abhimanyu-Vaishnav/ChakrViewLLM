# NR003: Checkpoint Comparison & Quantitative Audit

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  
**Hardware Platform:** NVIDIA RTX A1000 6GB Laptop GPU (CUDA 12.6, PyTorch 2.5.1+cu124)  

---

## 1. Verified Artifacts & Hash Registry

| Model Checkpoint | Description | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **`genesis_pretrain_best.pt`** | Stage-C Pretraining Best Checkpoint | 62,204,051 bytes | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` |
| **`genesis_pretrain_final.pt`**| Stage-C Pretraining Final Checkpoint | 20,733,533 bytes | `11c6f633492443cac11025bbc8f092d68393f4727fa7c1b18903c2e15dab21fa` |
| **`genesis_sft_best.pt`** | Genesis Conversational SFT Best Checkpoint | 20,733,097 bytes | `ebde2fb8e156ad1f74baa5245c3e4fd9fb35f3caa4e381b8261bbf3d0b2c28b4` |
| **`genesis_recovery_sft_best.pt`** | Balanced Recovery SFT Checkpoint | 20,733,719 bytes | `d31cc7c3c6dd02cdc22a0c8406469bd18044ea8fad5ca3c87c1487291c9944ea` |
| **`v4096_vocab.json`** | In-use candidate tokenizer vocabulary | 112,969 bytes | `fbde6847bb4e678edd2590908f472c0883d74c6abcb92c9b626269ea18a28adb` |
| **`v4096_merges.json`** | In-use candidate tokenizer merges | 76,310 bytes | `259f537b23e1186e719501f3994639cc537a2626316f32ffcec98d85a9274a89` |
| **`vocab_4096_canonical.json`** | Canonical vocabulary used to create Stage-C | 112,627 bytes | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |

---

## 2. Quantitative Comparative Evaluation

Evaluated under identical architectures ($d=256, L=6, H_q=8, H_{kv}=2$, RoPE, GQA, SwiGLU, parameter count = 5,177,600) and generation settings (temperature = 0.7, top-k = 40, max new tokens = 40):

| Evaluation Metric | 1. Pretraining-Only (`genesis_pretrain_best`) | 2. Genesis SFT (`genesis_sft_best`) | 3. Balanced Recovery SFT (`genesis_recovery_sft_best`) |
| :--- | :---: | :---: | :---: |
| **English Prose Loss / PPL** | 8.32 / 4,121.8 | **6.65 / 775.6** | 6.72 / 831.6 |
| **Hindi Prose Loss / PPL** | 9.80 / 18,208.8 | **4.30 / 74.2** | 6.75 / 854.8 |
| **Hinglish Dialogue Loss / PPL** | 9.24 / 10,341.9 | **6.71 / 824.6** | **6.59 / 732.7** |
| **SFT Dialogue Target Loss / PPL** | 8.05 / 3,160.1 | **3.52 / 34.1** | 4.29 / 73.5 |
| **Dialogue Response Formatting** | None (open-ended token scramble) | **High** (`"Hello! I am doing well..."`) | Moderate |
| **Prompt-Dependent Relevance** | 0.0% | **12.0%** (limited to exact memorized prompts) | 8.0% |
| **Token Diversity (Entropy)** | High (uncalibrated) | Moderate (collapses to memorized targets) | Moderate |
| **Infinite Attractor Loop?** | None | None | None |

---

## 3. Key Forensic Discovery

1. **Vocabulary Permutation Disconnect:**
   - The Stage-C binary tokenized shards on disk (`data/tokenized/stage_c/train/*.bin`) were pre-tokenized against `data/experiments/vocab_4096` (SHA-256 `db4d6119...`).
   - However, `train_pretraining.py` loaded `experiments/tokenizer/candidates/v4096` (SHA-256 `fbde6847...`).
   - Between these two 4,096-token dictionaries, **3,677 token IDs are permuted**. For example, token ID `265` represents `'2020'` (double spaces) in the canonical tokenizer, but represents bytes `e0 a4 be` in the candidate tokenizer.
   - Consequently, during Stage-C pretraining, the neural model learned cross-entropy predictions over randomly permuted subwords rather than coherent English/Hindi grammar.
2. **Impact on SFT Fine-Tuning:**
   - During SFT, `train_sft.py` encoded plain-text prompts from scratch using `candidates/v4096`.
   - The model was forced to learn language syntax entirely from the 1,476 SFT examples from scratch in 5 epochs, explaining why it regurgitated memorized response strings without input-dependent comprehension.
