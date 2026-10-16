# Training Readiness Decision

## Batches 1341–1380: Objective Assessment of ChakrMicro Re-Training Feasibility

### 1. Training Readiness Checklist

| Precondition | Status | Evidence & Verification Path |
|---|---|---|
| **Checkpoint Loads Reliably** | **PASS** | `training_checkpoint_step_10000.pt` loaded with 0 missing/unexpected keys. |
| **Tokenizer Compatibility** | **PASS** | Vocab 4096 perfectly aligned with embedding dimension [4096, 192]. |
| **Leakage-Controlled Dataset** | **PASS** | 52 documents strictly partitioned into train (25 docs) and test (27 docs). |
| **Baseline Metrics Established** | **PASS** | PPL 158.07, Loss 5.063, Top-1 12.85%, Top-10 40.39% benchmarked reproducibly. |
| **Causal Invariance Verified** | **PASS** | Verified 0 future token leakage in [test_batch1341_1380_lm_evaluation.py](file:///D:/Project/ChakrView/tests/test_batch1341_1380_lm_evaluation.py). |
| **Hardware & Compute Compatibility** | **PASS** | Local CPU execution achieves ~9,000 tokens/sec. 1,000 steps of batch size 4 takes < 2 minutes on CPU. |
| **Specific Training Hypothesis** | **PASS** | Formulated below: Continual pre-training on technical domain shards. |

---

### 2. Decision: Training is Justified but Deferred to Controlled Next Milestone

**Status:** **CONDITIONAL GO FOR BATCHES 1381+**

The empirical evidence shows that:
1. ChakrMicro is **not a random weight model**: it achieves $158.07$ perplexity on held-out text vs $754.93$ for unigram and $4096$ for random.
2. However, with an initial perplexity of $158.07$, the model exhibits high repetition under greedy decoding and has not yet converged to low loss on technical domain documentation.
3. Therefore, continual pre-training on domain technical shards is scientifically justified. In accordance with the Phase 7 instructions of this mission, **training must not be executed hastily in this milestone**. Instead, the complete training protocol is specified below and queued for Batch 1381+.

---

### 3. Formal Training Proposal for Batch 1381+

#### 3.1. Hypothesis
Continual pre-training of `ChakrMicro` on the 25 training-split technical documents for 500 optimization steps (learning rate $1\times 10^{-4}$ with cosine decay, AdamW) will reduce held-out validation loss from $5.0630$ to $< 4.5000$ (Perplexity $< 90.0$), without increasing loss on synthetic anchor sequences by $> 0.20$.

#### 3.2. Dataset & Split Specification
- **Training Set:** 25 documents in `experiments/batch1341_1380/data/corpus_lm_split_manifest.json` (`train_split`).
- **Validation / Held-Out Test Set:** 27 documents in `experiments/batch1341_1380/data/corpus_lm_split_manifest.json` (`test_split`).
- **Token Count:** ~3,200 training tokens; sequence length 128 tokens; effective batch size 4 sequences (512 tokens/step).

#### 3.3. Training Hyperparameters
- **Base Checkpoint:** `training_checkpoint_step_10000.pt` (read-only input).
- **Target Checkpoint:** `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_step_10500.pt` (new path, never overwriting canonical).
- **Optimizer:** `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)`
- **Gradient Clipping:** Max norm 1.0.
- **Warmup:** 50 steps; Cosine decay to $1\times 10^{-5}$ at step 500.
- **Compute Budget:** CPU execution, estimated 90 seconds total run time.

#### 3.4. Validation & Early Stopping Criteria
- Evaluate validation perplexity every 50 steps.
- Early stopping triggered if validation loss fails to decrease for 3 consecutive evaluations (150 steps).
- Checkpoint accepted only if held-out test PPL $< 90.0$ and top-1 accuracy $> 18.0\%$.
