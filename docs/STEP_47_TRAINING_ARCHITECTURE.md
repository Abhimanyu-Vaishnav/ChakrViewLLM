# STEP 47 — TRAINING ARCHITECTURE SPECIFICATION

## Training Readiness & First Learning Loop Phase

**Date:** 2026-09-30  
**Phase:** Step 47 Architecture  
**Status:** Approved Architectural Specification  

---

## 1. Existing Training Infrastructure

The ChakrView repository already includes foundational pre-training primitives:
- **Neural Core (`ChakrMicro`):** 3,443,136 parameters, 6 layers, 192 d_model, 6 heads, 512 context window, tied LM head (`chakrview/brain/model.py`).
- **Tokenizer (`BPETokenizer`):** 4,096 vocabulary byte-pair encoding (`chakrview/tokenizer/`).
- **Token Shards:** Validated binary token shards in `data/tokenized/train/` and `data/tokenized/val/`.
- **Dataset Streaming:** `StreamingTokenDataset` (`chakrview/training/dataset.py`) for sequential disk-to-tensor iteration.
- **Collator:** `CausalLanguageModelingCollator` (`chakrview/training/collator.py`) enforcing $[B, T \le 512]$ batch dimensions.
- **Loss:** `CausalLoss` (`chakrview/training/loss.py`) cross-entropy next-token prediction with PAD exclusion.
- **Optimizer & Scheduler:** AdamW with decoupled weight decay for 2D matrices and linear warmup cosine decay scheduler (`chakrview/training/optimizer.py`).
- **Checkpoint Manager:** `CheckpointManager` (`chakrview/training/checkpoint.py`) with atomic writes.
- **Safety Checks:** `TrainingSafetyChecker` (`chakrview/training/safety.py`) with loss and gradient checks.

---

## 2. Missing Components & Identified Vulnerabilities

Prior to Step 47, the training primitives were not unified into a hardened, fail-closed learning loop:
1. **Safety Checks Bypassed in Core Loop:** `Trainer.train_step()` failed to invoke `TrainingSafetyChecker.check_loss()` and `check_gradients()`.
2. **Gradient Telemetry Missing:** Gradient norm was not measured or tracked across steps.
3. **Unvalidated Checkpoint Typology:** Checkpoints lacked explicit `checkpoint_type="training"` classification, allowing inference checkpoints to be erroneously loaded into `Trainer.resume()`.
4. **Missing Manifest & Tokenizer Fingerprints:** Checkpoint payloads did not store or validate `tokenizer_checksum` or `dataset_manifest_hash`.
5. **No Unified Learning Loop Runner:** No executable script verified a complete, reproducible CPU training loop from shards to loss decrease to checkpoint resume.

---

## 3. Exact Step 47 Scope

Step 47 hardens the training pipeline to establish the first verifiable learning loop:
1. **Harden `CheckpointManager` (`chakrview/training/checkpoint.py`):**
   - Save `checkpoint_type: str = "training"`.
   - Embed `tokenizer_checksum`, `dataset_manifest_hash`, parameter count (3,443,136), and model config.
   - Enforce pre-flight validation in `load()` and `validate_checkpoint_payload()`.
   - Reject inference checkpoints or corrupted payloads with `CheckpointCorruptionError`.
2. **Harden `Trainer` (`chakrview/training/trainer.py`):**
   - Check loss finiteness before `.backward()`.
   - Check gradient finiteness and calculate gradient norm before clipping and optimizer step.
   - Record `grad_norm` and `step_latency_ms` in `MetricsTracker`.
   - Enforce checkpoint compatibility validation in `resume()`.
3. **Micro-Training Validation Run:**
   - Execute a 10-step micro-training run on real token shards.
   - Verify loss decrease, non-zero gradient norm, parameter mutation on the test model instance, and atomic checkpoint save/resume.
4. **Baseline Weight Protection:**
   - Explicitly verify that canonical baseline model weights (`c5571...`) are untouched ($\Delta W_{baseline} = 0$).

---

## 4. Architectural Data & State Flow

```mermaid
flowchart TD
    subgraph Data Loading
        DS[StreamingTokenDataset] -->|Token Sequences| COL[CausalLanguageModelingCollator]
        COL -->|Batch: input_ids, target_ids| TRN[Trainer.train_step]
    end

    subgraph Forward Pass & Loss
        TRN -->|Forward Pass| MDL[ChakrMicro]
        MDL -->|Logits [B, T, 4096]| LOSS[CausalLoss]
        LOSS -->|Cross-Entropy Loss| SCK1[TrainingSafetyChecker.check_loss]
    end

    subgraph Backward & Optimization
        SCK1 -->|Finite Scaled Loss| BCK[loss.backward]
        BCK -->|Compute Gradients| SCK2[TrainingSafetyChecker.check_gradients]
        SCK2 -->|Grad Norm & Sanity| CLP[clip_grad_norm_]
        CLP -->|Clipped Gradients| OPT[AdamW Optimizer Step]
        OPT -->|Update Parameters| SCH[Cosine LR Scheduler Step]
    end

    subgraph Checkpointing & State Verification
        SCH -->|Trigger Interval| CKM[CheckpointManager.save]
        CKM -->|Atomic Write| CKPT[training_checkpoint.pt]
        CKPT -->|Validate & Resume| RES[Trainer.resume]
    end
```

---

## 5. Checkpoint Schema Specification

A valid Step 47 training checkpoint contains:
```python
{
    "checkpoint_type": "training",           # Must be "training" (never "inference")
    "step": int,                             # Training step counter
    "epoch": int,                            # Epoch counter
    "timestamp": str,                        # ISO-8601 UTC timestamp
    "model_state_dict": dict,                # ChakrMicro parameters
    "optimizer_state_dict": dict,            # AdamW state
    "scheduler_state_dict": dict,            # Cosine LR scheduler state
    "rng_state": dict,                       # Torch, NumPy, Python RNG states
    "model_config": dict,                    # ModelConfig parameters
    "parameter_count": 3443136,              # Unique parameter count
    "tokenizer_checksum": str,               # SHA-256 of active tokenizer
    "dataset_manifest_hash": str,            # SHA-256 of training dataset metadata
    "train_metrics": dict,                   # Loss, lr, grad_norm, latency
    "val_metrics": dict,                     # Validation loss and perplexity
}
```

---

## 6. CPU Resource Constraints & Determinism

- **Hardware:** Pure CPU execution (`torch.device("cpu")`).
- **Memory Footprint:** Bounded streaming dataset iteration; total process RSS $< 350\text{ MB}$.
- **RNG Seeding:** Explicit seeding via `set_seed(seed)` ensuring deterministic weight updates and loss trajectories given identical data order.
- **Fail-Closed Semantics:** Any violation of loss finiteness, gradient stability, or checkpoint compatibility halts execution immediately.

---

## 7. Explicit Non-Scope for Step 47

- **No full-epoch training run.**
- **No large-scale dataset ingestion.**
- **No replacement of the frozen baseline weights.**
- **No GPU requirements.**
- **No claim of linguistic intelligence or natural language comprehension.**
