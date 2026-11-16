# UCIP-03R: Phase Six & Seven — Tokenizer and Checkpoint Compatibility Audit

**Mission ID:** UCIP-03R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Date:** 2026-11-16  

---

## 1. Checkpoint Architecture & Tensor Verification

| Parameter / Dimension | Specification (`ChakrMicro`) | Released Checkpoint (`chakrview_ucip03_model.pt`) | Parent Checkpoint (`v0.11`) | Match Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 3,443,136 | 3,443,136 | 3,443,136 | **PASS (BIT-EXACT)** |
| **Layers (`n_layers`)** | 6 | 6 | 6 | **PASS** |
| **Heads (`n_heads`)** | 6 | 6 | 6 | **PASS** |
| **Model Dim (`d_model`)** | 192 | 192 | 192 | **PASS** |
| **Hidden Dim** | 512 | 512 | 512 | **PASS** |
| **Vocab Size** | 4,096 | 4,096 | 4,096 | **PASS** |
| **Max Seq Len** | 512 | 512 | 512 | **PASS** |
| **Weight Tying** | `embedding.weight == lm_head.weight` | Tied | Tied | **PASS** |

---

## 2. Tokenizer Artifact Audit

- **Tokenizer Artifact Directory:** `data/experiments/vocab_4096`
- **Release Directory Tokenizer:** `releases/ucip_03_conversation_candidate/vocab.json`, `merges.json`
- **Integrity Check:**
  - `data/experiments/vocab_4096/vocab.json` SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `releases/.../vocab.json` SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `data/experiments/vocab_4096/merges.json` SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `releases/.../merges.json` SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
- **Finding:** The tokenizer artifacts are 100% bit-exact and identical across all directories. There is **zero tokenizer mismatch**.

---

## 3. Logit & Generation Tracing

Traced generation of prompt `User: Say exactly: The sky is blue.\nAssistant:`:
- Step 1: Logits strongly favor whitespace ` ` ($p=61.6\%$) followed by capital initial letters.
- Step 2: Model predicts `'The '` ($p=29.7\%$).
- Step 3: Model predicts `'p'` ($p=71.3\%$).
- Step 4: Model predicts `'lan'` ($p=9.5\%$).
- Step 5: Model predicts `'ned '` ($p=25.3\%$).
- Step 6: Model predicts `'of '` ($p=25.9\%$).
- Step 7: Model predicts `'the '` ($p=17.0\%$).
- Step 8: Model predicts `'b'` ($p=11.4\%$) $\to$ `'old.'`.

### Root Cause in Weights:
The model has formed an associative attractor between the prompt template tokens `"User: ... \nAssistant:"` and high-frequency training fragments (`"Fortune favors the bold."` and `"A garden is a planned outdoor space..."`). It blends unigrams and bigrams from these proverbs into a pseudo-syntactic sentence (`"The planned of the bold."`) rather than conditioning on the user's explicit payload (`"The sky is blue."`).
