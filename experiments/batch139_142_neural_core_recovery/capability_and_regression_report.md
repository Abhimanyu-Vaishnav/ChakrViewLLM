# Capability and Regression Report: Batch 139–142

**Date:** 2026-10-13  
**Milestone:** Batch 139–142 Neural-Core Recovery  
**Focus:** Neural Core Diagnostic & Capability Characterization  

---

## 1. Real Capability Growth vs. Experimental Proxies

In continuing ChakrView development, we explicitly distinguish between:
1. **Infrastructure/Pipeline Verification:** Proving that data loaders, optimizers, learning rate schedules, and checkpoint serializers operate deterministically and without runtime error.
2. **Neural Representation Learning:** Demonstrating that the model parameters have acquired stronger, more generalizable language or reasoning representations that reduce held-out entropy on unseen text.

### Honest Assessment of Batch 139–142:
- **Infrastructure Verification:** **ACHIEVED.** The pipeline now correctly implements linear warmup + cosine decay, bit-exact resumption, multi-shard partitioned evaluation, and deterministic cross-entropy computation. Resumability of candidate checkpoints was independently verified ($L_{resumed} = L_{orig} = 3.0517$).
- **Neural Representation Learning:** **INCONCLUSIVE / FLAT.** 250 steps of fine-tuning (128k tokens) from an already converged 10,000-step baseline model arrested the regression seen in Batch 135–138 (reducing loss degradation from $+0.12$ to $+0.005$), but did NOT produce a net decrease in global test loss compared to the baseline.
- **Repetition Pathology:** Autoregressive greedy continuations across all candidate models continue to exhibit 3-gram repetition loops ($0.27$ to $0.55$) and repetitive phrasing (`"in the world of the world"`, `"in the first of the first"`).

---

## 2. Quantitative Metric Summary

| Dimension | Baseline (Arm A) | Replicated (Arm B) | Warmup Natural (Arm C) | Warmup Balanced (Arm D) |
|:---|:---:|:---:|:---:|:---:|
| **Val Loss** | 3.2441 | 3.3643 | 3.2534 | 3.2614 |
| **Global Test Loss** | 3.0721 | 3.1936 | 3.0896 | 3.0776 |
| **English Test Loss** | 3.1970 | 3.3059 | 3.2002 | 3.2062 |
| **Hindi Test Loss** | 3.0569 | 3.1654 | 3.0720 | 3.0586 |
| **Code Test Loss** | 2.7540 | 2.8584 | 2.7719 | 2.7526 |
| **3-gram Repetition** | 0.2778 | 0.4841 | 0.5556 | 0.4365 |
| **Unique Token Ratio** | 0.5391 | 0.3594 | 0.3359 | 0.4141 |
| **Checkpoint Resumable** | Yes | Yes | Yes (Verified) | Yes |

---

## 3. Confirmed Regressions & Unresolved Limitations

### A. Repetition & Mode Collapse in Small-Capacity Models
- With 3.44M parameters (`d_model=192`, 6 layers) and greedy decoding (`argmax`), small models naturally settle into cyclic attractors when prompted out of distribution.
- Fine-tuning on 128k tokens does not expand syntactic depth. In fact, short-horizon fine-tuning slightly worsens token diversity (baseline unique ratio $0.5391 \to$ Arm C $0.3359$).

### B. The 10k Baseline Pretraining Saturation
- The canonical Step 10,000 baseline was pretrained on ~5.1M tokens with cosine decay down to $2\times 10^{-5}$.
- Starting a "Stage D" continuation on the same Stage C corpus with only 100–250 steps is unable to discover superior minima; it either destabilizes the converged weights (if LR is too high, as in Arm B) or oscillates near the baseline basin (Arms C and D).
- **Core Lesson:** Pretraining cannot be accelerated via micro-batches of a few hundred steps. Meaningful improvement of the neural core requires either:
  1. A significantly larger token budget (e.g., 2,500–5,000 steps with proper warmup/cosine decay).
  2. A differentiated, high-quality instruction or curriculum corpus rather than repeated sampling of the same pretraining shards.

---

## 4. Preservation of Invariants

All core repository invariants were rigorously maintained:
- **Canonical Step 10,000 Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` remains unmodified (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Tokenizer Contract:** Vocabulary size 4,096; BPE model in `release/chakrview-v0.5.0/data/experiments/vocab_4096` unmodified.
- **Architecture Invariants:** 3,443,136 parameters; seed 42 weight hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` verified.
- **User Work Preserved:** Modified `chakrview/runtime/session.py` remains uncommitted and untouched.
- **Untracked Files:** All pre-existing untracked files remain intact.
