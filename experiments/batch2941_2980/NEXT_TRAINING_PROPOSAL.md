# Next Training Proposal — Batch 2941–2980
## Balanced SFT Pilot Specification (Awaiting Operator Authorization)

### 1. Executive Summary & Context

Batch 2901–2940 revealed that fine-tuning the 2.5M parameter ChakrMicro model on 100% instruction pairs with mandatory terminal `<EOS>` induced a severe regression in raw continuation mode: on 15 out of 28 evaluation prompts, the model immediately produced `<EOS>` or empty output. 

Batch 2941–2980 has prepared the data, architecture, checkpointing, and verification infrastructure for a **balanced 70/30 SFT experiment**:
- **280 Curated Instruction Examples** (70%): Across 7 verified categories (context extraction, structured JSON, short explanation, text transformation, summarization, context QA, abstention).
- **120 Pretraining Replay Examples** (30%): Natural document continuations drawn from `data/raw/` pretraining corpora, supervised with standard causal next-token cross-entropy and **no artificial terminal `<EOS>`**.

**Current Status:** All preparation, dataset splits, best-checkpoint saving mechanisms, and preflight gates have passed. **Zero training or optimizer steps have been performed.** This document outlines the proposed training protocol for operator review and authorization.

---

### 2. Proposed Training Hyperparameters

| Parameter | Proposed Value | Rationale |
| :--- | :--- | :--- |
| **Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | Canonical Step-10,000 baseline (SHA-256: `96357705a683482e...`) |
| **Dataset Size** | 400 examples (320 Train, 40 Val, 40 Test) | 70% instruction / 30% continuation replay |
| **Batch Size** | 4 | Balances gradient stability with CPU training throughput (~80 batches per epoch) |
| **Max Steps** | 160 steps | Exactly 2 full epochs over the 320-example training set |
| **Learning Rate** | `5.0e-5` | Same gentle LR as Pilot 100-step run, preventing catastrophic forgetting |
| **LR Schedule** | Cosine decay to `5.0e-6` with 10-step warmup | Smooth annealing to prevent late-stage destabilization |
| **Weight Decay** | 0.01 | Standard AdamW regularization |
| **Gradient Clipping** | 1.0 | Prevents gradient spikes during fine-tuning |
| **Validation Cadence**| Every 10 steps | Evaluates validation loss across all 40 held-out validation samples |
| **Checkpoint Cadence**| Every 20 steps (periodic) + Best-Validation Checkpoint | Periodic checkpoints saved alongside atomic `sft_best_checkpoint.pt` |
| **Device** | CPU (torch.float32) | Deterministic CPU training matching runtime environment |
| **Seed** | 42 | Full determinism |

---

### 3. Best-Checkpoint & Early Stopping Strategy

In Batch 2901–2940, validation loss achieved its minimum at **step 40** (val loss 2.8988) before overfitting (step 100 val loss 3.4211), but only step 25, 50, 75, 100 checkpoints were saved.

With the new atomic best-checkpoint engine implemented in `chakrview/training/sft_engine.py`:
1. Every 10 steps, validation loss is evaluated over the 40 validation samples.
2. If `val_loss < best_val_loss` (strictly less), the state is atomically written to `sft_best_checkpoint.pt`.
3. If validation loss does not improve for **40 consecutive steps (4 validation intervals)** after step 40, early stopping is triggered.
4. Ties in validation loss are deterministically resolved by preserving the earlier checkpoint.

---

### 4. Resource & Runtime Estimate

- **Platform:** Windows x64 (CPU inference/training)
- **Forward/Backward pass speed:** ~0.18s per sequence (sequence length ~64-128 tokens)
- **Batch Size 4 throughput:** ~0.72s per optimizer step
- **Validation (40 samples):** ~5.5s per evaluation
- **160 Steps Runtime:**
  - 160 training steps: ~115 seconds
  - 16 validation runs: ~88 seconds
  - Checkpoint disk I/O: ~15 seconds
  - **Total Estimated Training Time:** ~3.5 to 4.0 minutes

---

### 5. Success & Regression-Defense Criteria

The run will be qualified according to the following acceptance gates:

1. **Instruction Following Adherence:**
   - JSON format adherence on held-out test prompts >= 66% (>= 2/3).
   - Context-grounded extraction exact match >= 30% (baseline is 0%).
   - Abstention accuracy >= 50% when given unanswerable context questions.
2. **Raw-Completion Preservation (Anti-Regression Gate):**
   - Immediate `<EOS>` / empty generation rate on the 28 raw-completion prompts must drop from **53.6% (15/28)** back toward baseline levels (< 15%).
   - Initial generated token for continuation prompts must NOT be `<EOS>` (token id 1).
3. **Artifact Integrity:**
   - `sft_best_checkpoint.pt` strictly reconstructs model weights and yields identical deterministic greedy output.
   - Canonical Step-10,000 checkpoint and tokenizer files remain untouched and bit-identical.

---

### 6. Operator Authorization Command

Training is currently locked behind a mandatory execution gate flag (`execute_training=False`).
When the operator decides to authorize training, the command will be:

```powershell
.\.venv\Scripts\python.exe experiments/batch2941_2980/run_preflight.py --execute-authorized-training
```

*(Note: In accordance with Invariant 6, this command has NOT been run during Batch 2941–2980.)*
