# Candidate Comparison Report: Batch 2901–2940
## Comparative Empirical Evaluation Across All Saved SFT Checkpoints

### 1. Checkpoint Inventory and Best-Checkpoint Status

Inspection of `experiments/batch2821_2860/sft_pilot_run/` revealed:
- `sft_checkpoint_step_25.pt` (Step 25, Val Loss 4.2496)
- `sft_checkpoint_step_50.pt` (Step 50, Val Loss 4.0857)
- `sft_checkpoint_step_75.pt` (Step 75, Val Loss 4.1210)
- `sft_checkpoint_step_100.pt` (Step 100, Val Loss 3.9569)

#### Critical Audit Finding on Best Checkpoint:
- In the training history, the lowest validation loss was recorded at **Step 40** (`val_loss = 3.9523`, Perplexity `52.06`).
- However, because the trainer configured `save_interval = 25`, checkpoints were saved strictly at steps 25, 50, 75, and 100.
- **No separate `sft_best_val_checkpoint.pt` was configured or saved by `SFTExperimentEngine`.**
- Therefore, a checkpoint specifically capturing Step 40 weights does not exist on disk.
- In accordance with non-negotiable project invariants, **we do NOT reconstruct it by rerunning training**. We document this architectural observation and evaluate all actually existing checkpoints.

---

### 2. Comprehensive Metric Matrix Across All 5 Checkpoints

| Checkpoint Name | Step | File SHA-256 | Parameter Digest | Val Loss | SFT Exact Match (24) | Structured JSON (3) | Raw Empty (28) | Plain Empty (28) |
|---|---|---|---|---|---|---|---|---|
| **`canonical_step_10000`** | 10000 | `96357705a683...` | `c5bf628bd059...` | Baseline | 0/24 (0.0%) | 0/3 (0.0%) | **0/28 (0.0%)** | 0/28 (0.0%) |
| **`sft_step_25`** | 25 | `ba8285d2c62b...` | `7d28fff56c9b...` | 4.2496 | 0/24 (0.0%) | 0/3 (0.0%) | 5/28 (17.9%) | 0/28 (0.0%) |
| **`sft_step_50`** | 50 | `36b3ee8a7e1a...` | `e98505c820e8...` | 4.0857 | 0/24 (0.0%) | **2/3 (66.7%)** | 19/28 (67.9%) | 0/28 (0.0%) |
| **`sft_step_75`** | 75 | `26e7881e3b5a...` | `08321109945e...` | 4.1210 | 0/24 (0.0%) | **2/3 (66.7%)** | 20/28 (71.4%) | 0/28 (0.0%) |
| **`sft_step_100`** | 100 | `f911d081c8a6...` | `d084b6f66d0a...` | **3.9569** | 0/24 (0.0%) | **2/3 (66.7%)** | 20/28 (71.4%) | 0/28 (0.0%) |

---

### 3. Analysis: Which Candidate Performs Best?

1. **Step-25 Candidate**:
   - Least prone to the raw-completion EOS regression (only 5 empty outputs vs 20 at step 100).
   - However, has not learned the structured JSON priors (0/3 valid JSON).
2. **Step-50 Candidate**:
   - First checkpoint to acquire the structured JSON output prior (2/3 valid JSON).
   - Validation loss `4.0857`.
3. **Step-100 Candidate**:
   - Lowest overall validation loss on the saved checkpoints (`3.9569`, Perplexity `52.30`).
   - Achieved 2/3 valid JSON parsing.
   - Exhibits identical raw-completion regression to step 75.

**Verdict**:
Among the saved candidates, **`sft_checkpoint_step_100.pt` is the best overall valid candidate** due to having the lowest validation perplexity and having acquired the structured JSON syntax prior, but it suffers from severe unconditioned continuation regression.
