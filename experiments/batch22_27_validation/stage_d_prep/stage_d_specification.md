# CHAKRVIEW STAGE D INSTRUCTION-TUNING SPECIFICATION & DESIGN
**Document Version:** 1.0.0  
**Status:** DRAFT & VALIDATED READINESS  
**Target Architecture:** ChakrMicro v0.1 (3.44M parameters, vocab 4096, d_model 192, layers 6, heads 6)  
**Parent Foundation Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  

---

## 1. PURPOSE & PRINCIPLES
Stage D introduces supervised instruction-following capabilities to ChakrView while preserving its indigenous, CPU-first, and zero-external-model principles.
- **Pretrained Weights Preservation**: The Step 10,000 pretraining checkpoint remains frozen and immutable. All Stage D experiments output to dedicated `experiments/stage_d_sft/` directories.
- **Zero Leakage**: Strict partition segregation across training, validation, and held-out test/benchmark examples.
- **Truthful Provenance**: No fabricated datasets or simulated external LLM distillation. Training examples derive from validated instruction tasks grounded in verified corpus knowledge.

---

## 2. INSTRUCTION DATASET SCHEMA (JSONL)
Each record in the Stage D instruction corpus must conform to the following schema:
```json
{
  "example_id": "CHAKR-SFT-000001",
  "language": "en | hi | hi-en | code",
  "task_type": "qa | grammar_correction | summarization | code_completion",
  "split": "train | val | test",
  "prompt": "Explain what Pre-RMSNorm does in ChakrMicro.",
  "response": "Pre-RMSNorm applies root-mean-square normalization to inputs before self-attention and MLP blocks, providing numerical stability across forward passes without learned bias vectors.",
  "provenance": "data/validation/english.txt",
  "quality_status": "VALID"
}
```

### Schema Rules & Validation Gates:
1. `example_id`: Unique string identifier (`CHAKR-SFT-XXXXXX`).
2. `language`: Restricted to supported dialects (`en`, `hi`, `hi-en`, `code`, `sa`, `math`).
3. `split`: Mandatory partition assignment (`train`, `val`, `test`).
4. Non-empty string constraint: Both `prompt` and `response` must be non-empty, stripped strings.
5. Content length bounds: `len(prompt) >= 5` and `len(response) >= 2`.
6. Token budget: Combined token count `len(prompt_tokens) + len(response_tokens) + 2 <= 512` (max sequence length).
7. Reject criteria: Duplicate text, invalid unicode, unescaped control chars, empty answers.

---

## 3. SUPERVISED LOSS MASKING (PROMPT-RESPONSE)
During standard pretraining, cross-entropy is calculated over all tokens $y_t = x_{t+1}$.
In Stage D supervised fine-tuning, loss is computed **strictly on response tokens**, with prompt tokens masked using PyTorch's `ignore_index = -100`:

```python
# Sequence format: <BOS> + prompt_tokens + response_tokens + <EOS>
# Target alignment:
# Target for prompt tokens: -100 (ignored by nn.CrossEntropyLoss)
# Target for response tokens: response_token_ids + <EOS>

def build_sft_batch(prompt_ids, response_ids, max_seq_len=512):
    bos = [0]
    eos = [1]
    input_ids = bos + prompt_ids + response_ids
    # Shifted target:
    # positions 0 to len(prompt_ids) are masked with -100
    target_ids = [-100] * (len(bos) + len(prompt_ids) - 1) + response_ids + eos
    
    # Truncate or pad to max_seq_len
    ...
```

### Invariant:
Only the gradient signals from generating the response tokens and the terminal `<EOS>` token update the model weights. The model is never penalized for the prompt's token distribution.

---

## 4. BOUNDED PILOT EXPERIMENT PLAN
- **Starting Checkpoint**: Step 10,000 checkpoint (`96357705a683...`).
- **Initial SFT Steps**: 500 steps.
- **Batch Size**: 2 (sequence length 128 to 256).
- **Learning Rate**: `5e-5` with linear warmup (50 steps) and cosine decay to `5e-6` (conservative to avoid catastrophic forgetting).
- **Resource Budget**: CPU-only, 4 threads, runtime budget < 120 seconds, peak RSS < 500 MB.
- **Validation Evaluation**: Evaluated every 50 steps on held-out SFT validation split and pretraining benchmark probes to detect regression.
- **Rollback Condition**: If pretraining benchmark target loss regresses by > 15%, the pilot automatically aborts and restores Step 10,000 checkpoint.
