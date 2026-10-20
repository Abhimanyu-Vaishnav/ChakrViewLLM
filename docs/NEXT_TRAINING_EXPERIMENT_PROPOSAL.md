# PROPOSAL: NEXT TRAINING EXPERIMENT (v0.8)
## STATUS: PROPOSED — NOT EXECUTED; USER AUTHORIZATION REQUIRED

> [!IMPORTANT]
> **MANDATORY SAFETY COMPLIANCE**
> In strict accordance with Sprint constraints:
> - **NO TRAINING HAS BEEN RUN.**
> - **NO OPTIMIZER STEPS OR WEIGHT UPDATES WERE EXECUTED.**
> - Execution of this proposal requires explicit user authorization.

---

## 1. Executive Summary & Objective

- **Target Capability**: Single-Task Copy-Augmented Context Grounding & Syntactic Function Completion.
- **Empirical Problem Statement**:
  The canonical 3.44M parameter ChakrMicro checkpoint (Step 10,000) achieves **0/4 (0.0%)** on context-specific copying and **1/32 (3.1%)** on the broader Native Intelligence Suite.
  Token-level empirical investigation confirmed that while punctuation alone does not trigger `<EOS>` termination (e.g. `What is the capital of France?` continues with `\n\nStep-by-Step Solution:`), the model suffers from massive pretraining dataset attractor bias, continuing almost all queries with generic mathematical textbook solutions or Wikipedia-style lists rather than grounded answers.
- **Core Hypothesis**:
  A narrow, low-capacity supervised fine-tuning (SFT) run (200 steps, batch size 8, learning rate 5e-5 with cosine decay) using strict loss masking (computing loss exclusively on response tokens, masking prompt tokens with `-100`) will condition the cross-attention and residual representations on exact context copying without destroying pretrained language features.

---

## 2. Lineage & Base Artifacts

- **Base Architecture**: ChakrMicro (6 layers, d_model=192, 6 heads, d_ff=512, vocab=4096, context=512)
- **Trainable Parameters**: 3,443,136
- **Base Checkpoint**: `D:/Project/ChakrView/experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Tokenizer**: `data/experiments/vocab_4096/`
  - `vocab.json` SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `merges.json` SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`

---

## 3. Dataset Splits, Leakage Prevention & Token Alignment

### Schema:
```json
{
  "task_id": "copy_grounded_001",
  "prompt": "Context: Sensor delta-781 is offline.\nQuestion: Which sensor is offline?\nAnswer:",
  "completion": " delta-781<EOS>"
}
```

### Splits & Dimensions:
- **Training Set**: 320 synthetic context-copying pairs + 80 minimal Python function completions (Total: 400 samples).
- **Validation Set**: 50 held-out pairs with non-overlapping synthetic entities (e.g., disjoint UUIDs/hex codes).
- **Frozen Test Suite**: 32 tasks from `evaluation/intelligence_suite/intelligence_tasks.json` (strictly held out, zero n-gram overlap with training set).

### Leakage Prevention & Token Verification:
- Strict disjoint entity sets: All training synthetic IDs (e.g. `sensor-A1`, `device-X9`) are strictly disjoint from test IDs (e.g. `gamma-9082`, `device-8821`, `sensor-omega`).
- Tokenizer alignment verified: Synthetic identifiers tokenize into valid multi-token BPE fragments without unknown token `<UNK>` fallbacks.
- Exact n-gram overlap check (n=3) between training examples and `intelligence_tasks.json`.

---

## 4. Hyperparameters & Optimization Schedule

| Parameter | Proposed Value | Justification |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Standard decoupled weight decay |
| **Learning Rate** | `5e-5` | Small learning rate to preserve pretrained weights |
| **Weight Decay** | `0.01` | Mild regularization against overfitting |
| **Warmup Steps** | 20 | Linear warmup over first 10% of budget |
| **Max Steps** | 200 | Strict step budget cap (< 1 minute CPU wall clock) |
| **Effective Batch Size** | 8 | Batch size 4 with gradient accumulation steps = 2 |
| **Max Sequence Length** | 128 tokens | Bound memory footprint |
| **Loss Masking** | `ignore_index = -100` on prompt tokens | Gradients computed solely on response completion tokens |
| **Precision** | Float32 (CPU) | Deterministic, CPU-first reproducibility |

---

## 5. Resource Estimates (CPU Execution)

- **Platform**: Windows 11 / Intel Core i7 / CPU-only PyTorch
- **Model Parameters**: 3.44M FP32 (~14 MB)
- **Optimizer Memory**: ~28 MB
- **Activation Footprint**: ~8 MB
- **Total RAM**: < 200 MB
- **Estimated Throughput**: ~4.5 steps/second on CPU
- **Total Training Duration**: ~45 seconds for 200 steps

---

## 6. Verification Gates & Success Criteria

### Baseline Scores (Frozen Pretrained Step 10,000):
- **Intelligence Suite**: 1/32 (3.1%) passed.
- **Context Copying**: 0/4 (0.0%) passed.
- **Coding Suite**: 2/12 functional pass rate (5/12 syntax valid).

### Required Post-Training Success Thresholds:
1. **Context Copying**: $\ge 50\%$ (at least 2/4 on held-out synthetic test instances).
2. **Instruction Following**: $\ge 25\%$ exact match.
3. **Coding Syntax**: No regression on 12-task coding syntax check ($\ge 5/12$).
4. **General Text Generation**: No degeneration into infinite unigram loops.

---

## 7. Explicit Abort / Failure Conditions

Training must automatically halt and roll back if:
1. Training loss diverges ($\text{Loss} > 15.0$ or $\text{Loss} = \text{NaN}$).
2. Validation loss fails to decrease after step 60 (early stopping patience = 3 evaluations).
3. Parameter mutation norm $\|\Delta W\|_2$ exceeds threshold $5.0$.
4. Memory consumption exceeds 1.0 GB.

---

## 8. Exact Launch Command (Pending Authorization)

```powershell
# DO NOT RUN WITHOUT EXPLICIT USER AUTHORIZATION
python -m chakrview.train.sft_engine `
  --base-checkpoint "D:/Project/ChakrView/experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt" `
  --train-data "data/sft/context_grounding_train.jsonl" `
  --val-data "data/sft/context_grounding_val.jsonl" `
  --max-steps 200 `
  --lr 5e-5 `
  --batch-size 4 `
  --grad-accum 2 `
  --output-dir "artifacts/experiments/sft_v08"
```