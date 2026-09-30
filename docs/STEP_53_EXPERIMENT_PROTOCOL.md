# ChakrView Step 53: Controlled Training Experiment Protocol

- **Protocol Version**: 1.0.0
- **Status**: Ratified Experiment Protocol
- **Target Model**: `ChakrMicro` v0.1 (3,443,136 parameters, FP32, CPU-only)

---

## 1. Controlled Experiment Matrix

To evaluate the impact of progressive curriculum additions, we establish a 4-tier experimental matrix using identical architectures and hyperparameters:

| Experiment | Curriculum Scope | Sample Tiers Included | Training Steps | Learning Rate | Batch Size | Device |
|:---|:---|:---|:---|:---|:---|:---|
| **EXP-A** | Token & Format Stability | Level 0A + Level 0B | 250 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |
| **EXP-B** | Stability + Computation | Level 0A + 0B + 0C | 250 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |
| **EXP-C** | Stability + Comp + Programming | Level 0A + 0B + 0C + 0D | 500 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |
| **EXP-D (Full)** | Comprehensive Curriculum | Levels 0A, 0B, 0C, 0D, 1 | 500 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |

---

## 2. Invariants & Hyperparameters

1. **Architecture Immutability**:
   - `n_layers = 6`, `d_model = 192`, `n_heads = 6`, `d_ff = 512`, `vocab_size = 4096`, `seq_len = 512`.
   - Weights initialized deterministically from seed `42` (`instantiate_frozen_baseline`).
2. **Frozen Baseline Preserved**:
   - Baseline checkpoint hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` remains immutable ($\Delta W_{\text{baseline}} = 0$).
   - All training occurs on isolated experimental models.
3. **Optimizer**:
   - Pure CPU AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay = $0.01$).
   - Gradient clipping: `max_norm = 1.0`.
4. **Checkpointing**:
   - Atomic replacement (`.tmp` $\to$ `.pt`).
   - Saved under `artifacts/step53/checkpoints/`.

---

## 3. Dual Evaluation Protocol

Each trained checkpoint is evaluated across two distinct, non-overlapping benchmarks:

1. **Step-52 Capability Benchmark (Frozen Anchor)**:
   - 20 Level-0 deterministic tasks evaluated under identical greedy sampling (`seed = 42`).
   - Direct comparability against Step-52 baseline scores (0.0%).
2. **Held-Out Generalization Benchmark (New in Step 53)**:
   - 10 completely unseen prompts not present in the training set:
     - Unseen addition: `4 + 7 = ` $\to$ `11`
     - Unseen multiplication: `3 * 6 = ` $\to$ `18`
     - Unseen comparison: `9 > 4` $\to$ `True`
     - Unseen python function: `def double(x):\n    return ` $\to$ valid AST
     - Unseen python utility: `def is_negative(val):\n    return ` $\to$ valid AST
     - Unseen structured JSON: `{"mode": "debug", "active": ` $\to$ valid JSON
     - Unseen transform: `The opposite of cold is ` $\to$ `hot`
     - Unseen factual: `The capital of Japan is ` $\to$ `Tokyo`
     - Unseen instruction: `Repeat the word 'chakrview':\nOutput: ` $\to$ `chakrview`
     - Unseen list continuation: `items = [10, 20, 30, ` $\to$ valid list AST

---

## 4. Resource & Telemetry Logging

For every training run, log:
- Wall time and steps per second.
- Peak RSS memory footprint (must remain $< 256\text{ MB}$).
- Pre- and post-validation cross-entropy loss and perplexity.
- Parameter delta $L_2$ norm: $||\Delta W||_2 = \sqrt{\sum (\theta_t - \theta_0)^2}$.
