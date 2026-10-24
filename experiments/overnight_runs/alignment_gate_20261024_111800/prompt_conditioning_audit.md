# PROMPT-CONDITIONING ALIGNMENT AUDIT & CONTRACT SPECIFICATION

**Audit ID**: `alignment_gate_20261024_111800`  
**Execution Timestamp**: 2026-10-24T11:26:00+05:30  
**Target Repository**: ChakrView Neural Core (ChakrMicro v0.1)  

---

## 1. Executive Summary & Root Cause Investigation

This audit investigated the prompt-conditioning contract across dataset generation, training batch synthesis, and runtime inference/evaluation.

### 1.1 The Actual Root Cause: Confirmed
A confirmed mismatch existed between the training input encoding in `chakrview/training/sft_engine.py` and the inference input encoding in `chakrview/runtime/neural_runtime_adapter.py`:
- **Training Path**: In `SFTDataset.get_batch`, examples were encoded using `rec["prompt"]` directly (e.g., `"What is the capital city of Japan?"`).
- **Inference & Evaluation Path**: In `NeuralRuntimeAdapter.generate`, the user prompt was formatted using `format_conditioned_prompt(prompt, prompt_mode="plain_instruction")` (producing `"Instruction: What is the capital city of Japan?\nResponse:"`).
- **The Empirical Consequence**: Because the model was trained on unadorned questions, presenting the prompt prefix `"Instruction: ...\nResponse:"` at inference triggered the base pre-training prior (repeating `"\nStep-by-Step Solution:\nThe total number..."`), causing the model to fail all instruction-following evaluation benchmarks. When evaluated on the unadorned prompt, the model generated natural conversational completions.

---

## 2. Canonical Prompt-Conditioning Contract

To prevent any future divergence between training, validation, and evaluation, the prompt-conditioning contract is formally established as follows:

### 2.1 Supported Conditioning Modes
The canonical conditioning formatter is `chakrview.runtime.prompt_conditioning.format_conditioned_prompt`:
1. `raw_completion`:
   - Template: `{query}`
   - Use: Pre-training continuity, raw completions.
2. `plain_instruction`:
   - Template: `Instruction: {query}\nResponse:`
   - Use: General single-turn instructions, factual QA, arithmetic, structured formatting.
3. `context_qa`:
   - Template: `Context: {context}\nQuestion: {query}\nAnswer:` (or `Question: {query}\nAnswer:` if context is empty).
   - Use: Context extraction, reference resolution, grounded reading comprehension.

### 2.2 Data Path & Boundary Alignment

| Stage | Input Representation | Tokenization & Delimiters | Target Labels / Loss Masking |
| :--- | :--- | :--- | :--- |
| **Dataset Record** | JSON record with `prompt`, `response`, `prompt_mode`, and pre-serialized `formatted_prompt`. | Ground truth formatted prompt pre-verified against `format_conditioned_prompt`. | Ground truth response string. |
| **Training (SFTDataset)** | `rec["formatted_prompt"]` (or dynamically formatted via `format_conditioned_prompt`). | `[BOS] + prompt_tokens + response_tokens` | `[-100] * len(prompt_tokens) + response_tokens + [EOS]` |
| **Inference / Eval** | User query + optional context + `prompt_mode`. | `[BOS] + prompt_tokens` (where prompt is formatted via `format_conditioned_prompt`). | Autoregressive generation up to `max_new_tokens` or `[EOS]`. |

### 2.3 Special Token Invariants
- `BOS_ID = 0`: Prepend to all causal sequences.
- `EOS_ID = 1`: Appended to target response in training. Generation halts upon emitting token 1.
- `PAD_ID = 2`: Dynamic batch padding token.
- `IGNORE_INDEX = -100`: All prompt tokens (`len(prompt_tokens)`) and padding tokens are masked with -100. **Prompt tokens never contribute to gradient updates.**

---

## 3. Minimal Safe Fix Implemented

1. **[sft_engine.py](file:///d:/Project/ChakrView/chakrview/training/sft_engine.py)**:
   - Updated `SFTDataset.get_batch` to check for `rec["formatted_prompt"]` first.
   - If missing, it dynamically applies `format_conditioned_prompt(rec["prompt"], prompt_mode=rec.get("prompt_mode", "plain_instruction"), context=rec.get("context", ""))`.
   - Passes the aligned formatted prompt string to `self.encode_example`.
2. **[task_evaluator.py](file:///d:/Project/ChakrView/chakrview/evaluation/task_evaluator.py)**:
   - Added word-boundary regex matching for short entities ($\le 2$ characters, e.g., chemical symbols `"W"`, `"Fe"`, `"Cu"`) in `_score_factual_qa`. This prevents false-positive substring matches (such as `"show"` matching `"W"`).
3. **[test_task_evaluator.py](file:///d:/Project/ChakrView/tests/test_task_evaluator.py)**:
   - Added single-character boundary regression tests confirming `"W"` matches `"Tungsten is denoted by W."` but correctly rejects `"The total number of the show"`.
4. **[test_sft_prompt_alignment.py](file:///d:/Project/ChakrView/tests/test_sft_prompt_alignment.py)**:
   - Created a dedicated unit test suite verifying deterministic token and mask parity between `SFTDataset` and `NeuralRuntimeAdapter`.
