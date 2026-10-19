# SFT V3 PIPELINE COMPATIBILITY & TRACE REPORT
**Scope:** Data-to-Model Pipeline Trace, Collator Inspection & Forward Compatibility  
**Date:** 2026-10-19  
**Execution Context:** `experiments/sft_v3_pilot_readiness/`

---

## 1. Executive Verdict

### **VERDICT: PIPELINE COMPATIBILITY 100% VERIFIED**

The newly generated SFT V3 dataset ([experiments/sft_v3_dataset_builder/](file:///d:/Project/ChakrView/experiments/sft_v3_dataset_builder/)) has been traced end-to-end through the canonical ChakrView training infrastructure ([chakrview/training/sft_engine.py](file:///d:/Project/ChakrView/chakrview/training/sft_engine.py)).
- Tokenizer encoding, causal sequence alignment, dynamic batch collation, loss-masking, and model forward passes operate with zero errors or shape mismatches.
- A forward-only cross-entropy loss calculation on representative SFT V3 batches completed cleanly on CPU, yielding an initial batch loss of **4.8898** (perplexity: **132.93**).
- Zero backward steps, optimizer creation, or weight modifications occurred.

---

## 2. End-to-End Pipeline Trace

We traced representative examples from raw JSONL storage to tensor collation and cross-entropy supervision:

### Step 1: Storage and Schema Ingestion
- `train.jsonl` contains 200 records: 140 grounded structured extraction and 60 continuation replay.
- `SFTDataset` ingests the JSONL file, verifies file presence, validates that no `HELD_OUT_BENCHMARK` records exist, and parses each line into structured in-memory records.

### Step 2: Normalization and Delimiter Policy
- In `encode_example()`, strings pass through `normalize_prompt_response_strings()`:
  - Structured extraction: Prompts end with `\nResponse:`. Responses begin with `{"` (no leading whitespace). Trailing newlines and delimiters are preserved when `strip_prompt_delimiters=False`.
  - Replay continuation: Prompts end naturally without delimiters. Trailing whitespace is trimmed to prevent token boundary anomalies.

### Step 3: Causal Sequence Alignment & Loss Masking
The causal alignment helper ([chakrview/training/prompt_contract.py](file:///d:/Project/ChakrView/chakrview/training/prompt_contract.py)) builds:
```python
# Structured Instruction Record (append_eos=True):
input_seq  = [BOS] + prompt_tokens + response_tokens
target_seq = [-100] * len(prompt_tokens) + response_tokens + [EOS]

# Replay Continuation Record (append_eos=False):
input_seq  = [BOS] + prompt_tokens + response_tokens[:-1]
target_seq = [-100] * len(prompt_tokens) + response_tokens
```
- **Prompt Masking**: `target_seq[:len(prompt_tokens)]` is filled with `-100`, ensuring the loss function (`nn.CrossEntropyLoss(ignore_index=-100)`) computes gradients exclusively on the model's generated tokens.
- **Causal Prediction Alignment**: At position `i = len(prompt_tokens)` (the final prompt token), the target is `target_seq[i] == response_tokens[0]`. The next-token prediction mathematics is strictly preserved.

### Step 4: Batch Dynamic Padding & Tensor Collation
In `get_batch(batch_size=4)`:
- Records are dynamically sampled.
- Sequences are padded to `max_len = max(len(x) for x in batch_inputs)` using:
  - Input padding: `PAD_ID = 2`
  - Target padding: `IGNORE_INDEX = -100`
- Output tensors: `x_batch` (Shape: `[4, max_len]`, dtype: `torch.long`), `y_batch` (Shape: `[4, max_len]`, dtype: `torch.long`).
- All token IDs reside within the discrete vocabulary range `[0, 4095]`.

### Step 5: Model Forward Pass & Forward-Only Loss
- Model: `ChakrMicro` initialized with canonical hyperparameters (`d_model=192`, `n_layers=6`, `n_heads=6`, `hidden_dim=512`, `max_seq_len=512`). Total trainable parameters: **3,443,136**.
- Forward pass: `logits = model(x_batch)` produces tensor of shape `[4, max_len, 4096]`.
- Loss evaluation: `loss_fn(logits.view(-1, 4096), y_batch.view(-1))` measures cross-entropy strictly on active response tokens (`active_tokens = 166`).
- Initial loss on Step-10,000 base weights: **4.8898**.

---

## 3. Verified Safety Controls

| Safety Verification Gate | Required Behavior | Measured Result | Audit Verdict |
| :--- | :--- | :--- | :--- |
| **Model Weights Invariance** | Parameter tensors in memory must not change | Bit-identical before and after | **PASS** |
| **Checkpoint Invariance** | Base checkpoint file on disk must not change | SHA-256 identical before and after | **PASS** |
| **Canonical Directories** | No writes permitted to canonical release dirs | 0 writes to canonical paths | **PASS** |
| **Zero Backward Steps** | No `.backward()` calls permitted | 0 backward calls executed | **PASS** |
| **Zero Optimizer Steps** | No `.step()` calls permitted | 0 optimizer steps executed | **PASS** |
| **Held-out Split Isolation** | `test.jsonl` must never be read or opened | 0 accesses to test split | **PASS** |

---

## 4. Pipeline Trace Checklist

- [x] **Correct prompt/target concatenation**: Verified across all 7 categories.
- [x] **Correct prompt-loss masking**: `-100` spans precisely `len(prompt_tokens)`.
- [x] **Correct target supervision**: Active response tokens unmasked.
- [x] **Correct padding and attention**: Dynamic padding with `PAD_ID=2` and `target=-100`.
- [x] **Correct EOS placement**: Instruction targets end with `1` (`<|endoftext|>`), replay continuations do not.
- [x] **Truncation bounds**: Max sequence length is 157 tokens, well within the 512 context limit.
- [x] **Dtype and Shape parity**: `torch.long`, 2D batch tensors `(B, T)`.
- [x] **Vocabulary alignment**: All token IDs < 4096.
- [x] **Base checkpoint loading**: Loads `model_state_dict` without weight modification.
