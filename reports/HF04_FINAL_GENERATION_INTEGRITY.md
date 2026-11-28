# HF-04 — FINAL GENERATION STOPPING AND SPECIAL TOKEN CORRECTNESS AUDIT REPORT

**Mission Reference:** HF-04  
**Date:** 28 November 2026  
**Target Repository:** `AbhimanyuVaishnav/chakrview-conversational-alpha`  
**Staging Directory:** `artifacts/huggingface/chakrview-conversational-alpha/`  
**Project Workspace:** `D:\Project\ChakrView`  
**Audit Status:** **PASSED & SEALED FOR PRODUCTION DEPLOYMENT**

---

## 1. Executive Summary

Mission **HF-04** was executed as a mandatory, independent verification mission following the discovery of token ID `2` overload across pretraining, SFT, and canonical tokenizer specifications.

### Key Audit Outcomes:
1. **Source Code Line-by-Line Investigation:** Audited `special_tokens.py`, `data/experiments/vocab_4096/config.json`, `train_stage1.py`, `train_conversational_sft.py`, and `chakrview/inference_package.py`.
2. **Resolution of Token Semantics:** Proved conclusively that Token ID `2` is canonically defined as `<PAD>` in tokenizer configuration, but was empirically trained as the **Document / Turn End Delimiter** in both pretraining and SFT pipelines.
3. **Inference Layer Remediation:** Updated `chakrview/inference_package.py` and staged standalone `inference.py` to stop on `stop_token_ids = {1, 2}` while enforcing prompt-prefix invariance (padding inside input prompts will never trigger premature termination).
4. **Automated 10-Case Regression Test Suite:** Created `experiments/cv24h_release/test_hf04_generation_regression.py` and executed on both CPU (`.venv_hf_audit`) and GPU (`.venv_gpu`, NVIDIA RTX A1000 Laptop GPU). **10/10 tests passed on both platforms**.
5. **Package Synchronization:** Updated staged documentation (`README.md`, `INFERENCE_GUIDE.md`) and added `inference.py` to the Hugging Face staging directory.

---

## 2. Source of Truth & Token Semantics Investigation

### 2.1 The Canonical Tokenizer Contract
The canonical Byte-Level BPE tokenizer (`data/experiments/vocab_4096/`):
- `bos_token_id`: **0** (`<BOS>`)
- `eos_token_id`: **1** (`<EOS>`)
- `pad_token_id`: **2** (`<PAD>`)
- Base byte offset: `byte_offset: 3` (Token IDs `3` through `258` correspond to raw bytes `0x00` through `0xFF`).
- Merge tokens: Token IDs `259` through `4095`. Total vocabulary size: **4,096**.

### 2.2 The Empirical Training Invariant
Line-by-line inspection of training pipelines revealed why Token ID `2` is emitted at the end of generated sequences:
- **`train_stage1.py` (Pretraining DataLoader):**
  When packing documents into sequences, the pipeline executes:
  ```python
  tokens.append(2)  # EOS delimiter between pretraining documents
  ```
- **`train_conversational_sft.py` (Conversational SFT DataLoader):**
  When encoding multi-turn conversation dialogues:
  ```python
  # Assistant response sequence ends with token 2:
  resp_tokens = tokenizer.encode(turn["content"]) + [2]
  # Padding sequence to fixed length:
  input_ids = input_ids + [0] * pad_len  # or [2] * pad_len
  labels = labels + [-100] * pad_len     # masked with -100
  ```
- **Finding:** Supervised learning explicitly trained the model to predict **Token ID `2`** when an assistant utterance or document ends. Token ID `1` was rarely or never supervised as an utterance termination signal during SFT.
- **Classification:** Token ID `2` is an **intentionally overloaded special token** behaving as `<PAD>` for tensor batching/loss masking and as `<turn_end>` for autoregressive generation.

### 2.3 Explicit Special Token Mapping Table

| Special Token | Canonical ID | Tokenizer String | Pretraining Role | SFT Role | Inference Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`BOS`** | `0` | `<BOS>` | Sequence start | Sequence start / Padding fallback | Safe prefix; does not halt generation |
| **`EOS`** | `1` | `<EOS>` | Canonical end | Secondary end | Halts generation immediately |
| **`PAD` / Delimiter** | `2` | `<PAD>` | Document delimiter | Utterance end delimiter & padding | Emitted by model to conclude response; halts generation immediately |
| **Input Padding** | `0` or `2` | N/A | Ignored / masked | Masked with `-100` | Protected prefix; never triggers generation stopping |

---

## 3. Inference Layer Generation Stopping Logic

To ensure deterministic stopping without premature termination, the inference generator (`chakrview/inference_package.py` and `inference.py`) implements:

1. **Context Headroom Invariant:**
   Prompts longer than `max_context - max_new_tokens` are safely trimmed from the left to guarantee space for `max_new_tokens`.
2. **Prompt-Prefix Protection:**
   Generation stopping is evaluated **strictly** on tokens emitted in the autoregressive loop (`step >= 0`). Any occurrence of token `2` or `0` within the initial `input_ids` tensor (whether padding, previous turn delimiters, or prompt text) is ignored by the stopping evaluator.
3. **Dual Stop Set:**
   ```python
   self.stop_token_ids = {1, 2}  # Canonical EOS (1) and Trained Delimiter (2)
   if next_token in self.stop_token_ids:
       stopped_on = f"EOS/Delimiter (token {next_token})"
       break
   ```
4. **Clean Detokenization:**
   When decoding for user display, the trailing delimiter (token `2` or `1`) is excluded from text presentation, while raw token arrays retain full fidelity in return metadata.

---

## 4. Automated 10-Case Regression Test Results

The automated regression test script `experiments/cv24h_release/test_hf04_generation_regression.py` was executed across two distinct environments:
- **Clean CPU Environment:** `.venv_hf_audit` (`torch==2.15.0+cpu`, isolated virtual environment)
- **Production GPU Environment:** `.venv_gpu` (`torch==2.14.1+cu126`, CUDA 12.6, NVIDIA RTX A1000 6GB Laptop GPU)

### Regression Test Matrix

| Test ID | Test Description | CPU Status | GPU Status | Verification Note |
| :--- | :--- | :--- | :--- | :--- |
| **Test 1** | BOS at prompt start | **PASS** | **PASS** | Token `0` allowed at prompt start; generates correctly and halts on EOS |
| **Test 2** | EOS termination (token 1) | **PASS** | **PASS** | Token `1` terminates loop immediately; remaining tokens not emitted |
| **Test 3** | PAD/delimiter termination (token 2) | **PASS** | **PASS** | Token `2` terminates loop immediately as empirical turn end |
| **Test 4** | PAD inside input prompt padding positions | **PASS** | **PASS** | Tokens `[2, 2, 0]` inside prompt prefix do NOT stop generation prematurely |
| **Test 5** | BOS emitted during generation | **PASS** | **PASS** | Emitting token `0` during generation does not halt when `stop_token_ids={1, 2}` |
| **Test 6** | Empty input handling | **PASS** | **PASS** | Empty string `""` falls back to `[BOS_TOKEN_ID]` and produces generation |
| **Test 7** | Variable length independent generation | **PASS** | **PASS** | Short sequences (2 tokens) and long sequences (5 tokens) halt independently |
| **Test 8** | Maximum context boundary | **PASS** | **PASS** | Trimming + generation respects `max_context=256` tokens exactly without index error |
| **Test 9** | Generation after previous assistant turn | **PASS** | **PASS** | Multi-turn prompt containing previous turn delimiters handles generation seamlessly |
| **Test 10** | No premature termination | **PASS** | **PASS** | Sequence of 7 non-delimiter tokens followed by token `2` generates exactly 8 tokens |

**Result:** **10/10 PASSED on CPU and 10/10 PASSED on CUDA.**

---

## 5. Hugging Face Staging Artifact Inventory & Checksums

The staging directory `artifacts/huggingface/chakrview-conversational-alpha/` contains all verified standalone files for the model release:

| Filename | Size (Bytes) | SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- |
| `bpe.py` | 5,417 | `01d9713d8ae3aaa63e6dca9bce24db0d77684c42a61898294fb6cb53e5ea0f75` | Tokenizer BPE training and merge logic |
| `bpe_engine.py` | 4,077 | `149fa67c4c9326b2c0f350fbd4c795e4ce132a4fa5147a659b36e16f739c1465` | Tokenizer runtime engine |
| `bytes.py` | 2,951 | `44189cb50516387a9cacf1752b38e0801a35bb136b64216fe44b07bc8f674303` | Tokenizer byte-to-unicode mappings |
| `config.json` | 477 | `756c323ab366713df21194a66d63d1f97faca7d876d66c62e53760302c8acc50` | Model architecture hyperparameters |
| `decoder.py` | 2,757 | `75fb48df1e853a0b0c22f58a4cf7592330a2daa4d6f70c678d0b7d88f30d4414` | Tokenizer decoding routines |
| `encoder.py` | 2,755 | `abd85026ef3fc54f256aeae10b9dcea690b440ddef22ed2aaef5f28a82492871` | Tokenizer encoding routines |
| `inference.py` | 8,535 | `7dff2613af38641e166d24d10d5cf3271024c702b49820e3c766a95568c18e04` | Standalone verified generator & chat CLI |
| `INFERENCE_GUIDE.md` | 3,250 | `ea8827c89fb3ed226b42325a6300f931aa281dc6fe45fa20afe66e67db1a58d4` | Detailed execution and sampling instructions |
| `merges.json` | 76,299 | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Canonical tokenizer merge dictionary |
| `model.safetensors` | 150,008,528 | `88c7ecd817648356a3e399872682c20a59bebc02873f7af6ae7e795cf8f66724` | 38M parameter tied safetensors weights |
| `MODEL_LIMITATIONS.md`| 2,848 | `b322709577f8994ed7d8157d17bdca3b4eb8a13d356eca2abb3acbede1aabf5d` | Candid limitations & failure modes |
| `modeling_chakrgenesis.py`| 7,376 | `da08f2bd1690158bf926afde7db193682ce7cd31f089739de87cc39000cf260f`| Standalone PyTorch ChakrGenesis38M model |
| `README.md` | 4,845 | `de097a801205cceddb68fbb119a248105a81886a4c9aa3e3a738c3402e730bae` | Complete HF Model Card with stopping code |
| `special_tokens.py` | 1,068 | `382f01704738b959f8cb2b1c4e52e35a02d2b5ddaa700afa54b70dc35ac77cd9` | Canonical special tokens module |
| `tokenization_chakrview.py`| 5,588| `5a99382c7e692ac907205bb186b1bbd495f8f7cc647e1af3cc739c16eed1a06b`| Tokenizer standalone loader |
| `tokenizer_config.json`| 1,130 | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589`| Tokenizer configuration file |
| `vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Canonical vocabulary mapping (4,096 tokens) |

---

## 6. Audit Conclusion & Publication Gate Recommendation

All special token ambiguities and generation stopping mechanics have been completely investigated, resolved, verified, and regression-tested.

### Gate Verdict: **READY FOR UPLOAD**
- Staged weights match exact conversational checkpoint weights.
- Staged tokenizer matches exact canonical vocabulary SHA-256 (`db4d...`).
- Staged architecture supports clean zero-dependency loading via PyTorch and safetensors.
- Staged generation stopping operates reliably with prompt-padding protection across CPU and GPU hardware.
- No uploads to Hugging Face have been performed in accordance with HF-04 boundaries.
