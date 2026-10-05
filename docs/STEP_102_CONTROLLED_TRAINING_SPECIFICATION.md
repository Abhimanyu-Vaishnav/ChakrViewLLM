# ChakrView Step 102: Controlled Stage B Curriculum Training Specification

- **Document Identifier**: `STEP_102_CONTROLLED_TRAINING_SPECIFICATION.md`
- **Milestone Designation**: Step 102 (Controlled Scientific Neural Training Experiment)
- **Status**: RATIFIED SPECIFICATION
- **Date**: October 5, 2026
- **Auditor**: Antigravity Core Cognitive Engineering
- **Baseline Invariant**:
  - Parameters: `3,443,136`
  - Canonical Weight SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
  - Canonical Invariant Rule: $\Delta W_{\text{baseline}} \equiv 0$ across all operations.

---

## 1. Experimental Hypothesis

> **Scientific Hypothesis**:  
> Training a separate, initialized candidate instance of `ChakrMicro` for 500 optimization steps on the **Stage B** curated corpus (effective batch size: 4 sequences = 1,024 tokens per step, sequence length: 256 tokens, cosine LR decay: $1.5 \times 10^{-3} \to 1.5 \times 10^{-4}$) will:
> 1. Measurably reduce held-out validation loss from $\approx 8.35$ to $\le 4.80$.
> 2. Compress held-out validation perplexity from $\approx 4,231$ to $\le 120.0$.
> 3. Generalize to the untouched Stage B test split with test perplexity $\le 130.0$ and generalization gap $|\mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}| \le 1.25$.
> 4. Improve top-5 accuracy on the frozen 40-probe Syntactic Probe Benchmark from the baseline level of $5.0\%$ to $\ge 25.0\%$.
> 5. Maintain zero gradient explosions (all $\|\mathbf{g}\|_2 \le 1.0$), zero NaNs/Infs, and zero weight mutations of the canonical baseline.

---

## 2. Experimental Data & Split Integrity (Phase 0 Audit Verified)

All splits are physically disjoint, pre-sharded into uint16 binary files, and cryptographically verified:

| Split | Directory Path | Token Count | Documents | Shard Files & SHA-256 Prefix | Role in Step 102 |
|---|---|---|---|---|---|
| **Train** | `data/tokenized/stage_b/train/` | 405,244 | 7,258 | `shard_00000.bin` (`695f22...`)<br>`shard_00001.bin` (`691e93...`) | Gradient optimization only |
| **Validation** | `data/tokenized/stage_b/validation/` | 48,527 | 891 | `shard_00000.bin` (`c69f7e...`) | Periodic evaluation (every 50 steps) |
| **Test** | `data/tokenized/stage_b/test/` | 47,882 | 853 | `shard_00000.bin` (`1e3bce...`) | Final evaluation ONLY (untouched during training) |

### Strict Boundary Verifications:
- **Zero Evaluation Leakage**: Train loader never accesses validation or test shards.
- **Zero Cognitive Scaffolding**: Persistent Project Brain, investigation loops, episodic recall, and strategy evolution are completely uninstantiated during neural training.
- **Strict Seed Control**: `seed=42` pins PyTorch CPU generator, NumPy RNG, and Python random generator.

---

## 3. Measured Canonical Baseline (Phase 1 Established)

Evaluated under the exact protocol on the frozen canonical baseline (`c5571c9...`):

| Evaluation Metric | Measured Baseline Value | Note / Interpretation |
|---|---|---|
| **Validation Loss** (Stage B Val, 188 batches) | **8.3503** | Near-uniform vocab distribution ($\ln 4096 \approx 8.318$) |
| **Validation Perplexity** | **4,231.49** | Complete epistemic uncertainty |
| **Test Loss** (Stage B Test, 186 batches) | **8.3508** | Identical baseline level on test set |
| **Test Perplexity** | **4,233.59** | Complete epistemic uncertainty |
| **Next-Token Top-1 Accuracy** (Validation stream) | **0.117%** ($6 / 5,120$) | Near random baseline ($1/4096 \approx 0.024\%$) |
| **Next-Token Top-5 Accuracy** (Validation stream) | **0.156%** ($8 / 5,120$) | Near random baseline ($5/4096 \approx 0.122\%$) |
| **Syntactic Probe Top-1 Accuracy** (40 probes) | **5.00%** ($2 / 40$) | 2 accidental top-1 matches (comma delimiters) |
| **Syntactic Probe Top-5 Accuracy** (40 probes) | **5.00%** ($2 / 40$) | Delimiters only; 0% on Python syntax / logic |
| **Syntactic Probe Mean Log Prob** | **-8.3773** | Corresponds to $e^{-8.38} \approx 0.00023$ per probe token |

---

## 4. Frozen Syntactic Probe Benchmark (Phase 2 Frozen)

- **Artifact Path**: `tests/fixtures/step102_syntactic_probes.json`
- **Probe Count**: 40 deterministic cloze probes across 4 functional categories:
  1. `python_syntax` (10 probes): Keywords, function signatures, imports, colon syntax (`def `, `return `, `in `, `import `).
  2. `delimiters` (10 probes): List brackets, tuple parentheses, dictionary braces, commas (`]`, `)`, `}`, `, `).
  3. `control_flow` (10 probes): Binary logical operators, assignment, identity comparison (`and `, `or `, ` = `, `is `).
  4. `natural_language` (10 probes): Multi-word lexical patterns, structural headers (`Once upon a`, `Table 1: Summary of results`).
- **Scoring Protocol**: Evaluated strictly on the output distribution of the final prompt token ($[1, T, V]$ at index $T-1$) using `SyntacticProbeEvaluator`.

---

## 5. Training Protocol & Checkpoint Governance (Phase 3 & 4)

- **Hardware Execution**: Pure CPU (`torch.device("cpu")`), multi-threaded execution.
- **Model Invariant**: `ChakrMicroTransformer` (3,443,136 parameters, 6 layers, 256 hidden dimension, 8 heads, 1024 FFN).
- **Hyperparameter Profile**:
  - Total Steps: `500`
  - Sequence Length ($T$): `256`
  - Micro-Batch Size ($B$): `2`
  - Gradient Accumulation Steps: `2` (Effective Batch Size = 4 sequences = 1,024 tokens)
  - Optimizer: Decoupled AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay = $0.01$ on 2D matrices, $0.0$ on biases/norms)
  - Learning Rate Schedule: Cosine decay with linear warmup
    - Peak LR: $1.5 \times 10^{-3}$
    - Min LR Floor: $1.5 \times 10^{-4}$
    - Warmup Steps: `25` steps
  - Gradient Clipping: Maximum L2 norm = `1.0`
- **Checkpointing & Isolation**:
  - Target Directory: `artifacts/step102_stage_b_curriculum/checkpoints/`
  - Save Cadence: Every 100 steps + Step 500 final.
  - History Retention: Last 3 checkpoints retained.
  - Isolation Rule: Training writes **only** to `artifacts/step102_stage_b_curriculum/`. Canonical baseline weights are never overwritten.

---

## 6. Target Metrics vs. Scientific Verdict Thresholds

| Metric | Measured Baseline | Target Criteria | Failure Criteria |
|---|---|---|---|
| **Validation Loss** | 8.3503 | $\le 4.80$ | $\ge 6.00$ or diverging |
| **Validation Perplexity** | 4,231.49 | $\le 120.0$ | $\ge 400.0$ |
| **Generalization Gap** ($|\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}|$) | 0.00 | $\le 1.25$ | $\ge 2.00$ (severe overfitting) |
| **Syntactic Probe Top-5 Accuracy** | 5.00% | $\ge 25.0\%$ | $< 10.0\%$ |
| **Syntactic Probe Top-1 Accuracy** | 5.00% | $\ge 15.0\%$ | $< 5.0\%$ |
| **Numerical Health** | 100% Finite | 0 NaNs, 0 Infs | Any NaN / Inf detected |
| **Canonical Baseline Integrity** | `c5571c...` | Exact Bit-Match | Any mismatch ($\Delta W \neq 0$) |

---

## 7. What This Milestone Must NOT Claim

Even upon total achievement of all target criteria:
- **No AGI, consciousness, or broad world reasoning** will be claimed.
- **No open-domain conversational mastery** will be claimed.
- **No zero-shot general instruction following** will be claimed.
- A positive result proves strictly that **ChakrMicro exhibits genuine next-token syntactic representation learning on its training distribution**.
