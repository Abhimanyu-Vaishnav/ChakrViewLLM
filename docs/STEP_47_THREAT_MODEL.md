# STEP 47 — THREAT MODEL & TRAINING SAFETY SPECIFICATION

**Date:** 2026-09-30  
**Phase:** Step 47 — Training Readiness & First Learning Loop Phase  
**Component:** Training Pipeline & Checkpoint System  

---

## 1. Threat Matrix

| Threat ID | Threat Description | Attack Vector / Trigger | Severity | Mitigation Strategy | Enforcement Mechanism |
|---|---|---|---|---|---|
| **TRN-01** | Silent Gradient Poisoning via NaN/Inf Loss | Pathological input sequence, numerical underflow/overflow in logits producing NaN/Inf loss | **Critical** | Pre-Backward Finiteness Assertion | `TrainingSafetyChecker.check_loss()` verifies loss is finite real number before calling `scaled_loss.backward()`; raises `NumericalInstabilityError` |
| **TRN-02** | Parameter Corruption via Exploding / Corrupted Gradients | Exploding activations causing gradient tensors to contain NaNs or Infs | **Critical** | Pre-Step Gradient Inspection & Norm Clipping | `TrainingSafetyChecker.check_gradients()` checks all parameter gradients for NaN/Inf and exploding norm before `optimizer.step()`; clips gradients to ceiling |
| **TRN-03** | Inference Checkpoint Resumed as Training Checkpoint | Caller passes an inference checkpoint (`checkpoint_type="inference"` or raw `model_state_dict`) into `Trainer.resume()` | **High** | Explicit Checkpoint Typology Enforcement | Checkpoint metadata embeds `checkpoint_type="training"`; `resume()` fails closed if type != "training" or optimizer state is missing |
| **TRN-04** | Overwriting Frozen Baseline Weights ($\Delta W_{baseline} \neq 0$) | Training loop saves updated weights to baseline model path or mutates baseline reference | **Critical** | Strict Baseline Separation & Pre/Post Hash Verification | Baseline model weights verified against `c5571...`; training operates strictly on isolated model instances; experiment weights saved to isolated dirs |
| **TRN-05** | Tokenizer / Dataset Mismatch on Resume | Training resumed using a tokenizer with different vocabulary or merges | **High** | Cryptographic Tokenizer Fingerprint in Checkpoint | Checkpoints store `tokenizer_checksum`; validated against active tokenizer before resume |
| **TRN-06** | Checkpoint Corruption via Process Crash / Partial Write | Power failure, SIGKILL, or disk full during `torch.save()` | **High** | Atomic Write via Temporary File & Replace | Write to `.pt.tmp`, sync, and invoke `os.replace` to final `.pt` atomically |
| **TRN-07** | Out-of-Bounds Token IDs | Corrupted dataset shard contains token IDs outside vocabulary range $[0, 4095]$ | **High** | Pre-Forward Token Bounds Verification | `TrainingSafetyChecker.verify_token_ids()` asserts all elements in `input_ids` and `target_ids` are within $[0, 4095]$ or equal `pad_token_id` |
| **TRN-08** | Batch Context Window Overflow | Batch sequence length $T > 512$ | **High** | Collator & Model Bounds Check | `CausalLanguageModelingCollator` asserts $T \le 512$ with fail-closed exception; silent truncation strictly prohibited |
| **TRN-09** | Non-Deterministic Training Trajectory | Inconsistent random seeds across Python, NumPy, and PyTorch | **Medium** | Unified Seed & RNG State Preservation | `set_seed()` synchronizes all RNGs; RNG states serialized into checkpoints and restored on resume |
| **TRN-10** | Architecture Mismatch on Checkpoint Load | Checkpoint from different layer count or dimension loaded into ChakrMicro | **Critical** | Structural & Parameter Count Validation | Verify unique parameter count equals exactly 3,443,136 and tensor shapes match `ModelConfig` |

---

## 2. Invariants Enforced During Step 47

1. **Frozen Baseline Immutability:**
   - The canonical baseline model weights digest remains:
     `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
   - Verified before and after all training experiments.
2. **Experimental Weight Evolution:**
   - For trained test models, parameter updates must be observable ($\Delta W_{experiment} > 0$).
   - Trained checkpoint SHA-256 must differ from baseline hash.
3. **Fail-Closed Safety:**
   - Any NaN/Inf in loss or gradients halts execution immediately.
   - Any corrupted or incompatible checkpoint aborts resumption immediately.
