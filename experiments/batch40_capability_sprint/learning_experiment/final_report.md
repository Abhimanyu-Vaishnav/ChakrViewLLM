# Learning Experiment Report: Batch 40 Controlled Continuation Learning

**Run ID:** `run_batch40_continuation_learning_20steps`  
**Date:** 2026-10-11  
**Target:** Controlled continuation learning on ChakrMicro v0.1 using deduplicated Stage B candidates.

---

## 1. Experiment Objective & Protocol

This experiment measures whether ChakrMicro v0.1 (3,443,136 parameters) can undergo controlled continuation training on newly extracted Stage B candidates without corrupting model state, causing loss divergence, or damaging canonical baseline integrity.

- **Initial Checkpoint**: Step 10,000 canonical pretraining checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Dataset**: 39 training examples, 7 held-out validation examples (`val.jsonl`), strictly partitioned at the source-document boundary.
- **Safety**: `SFTBatchSafetyGuard` enforced on all input/target batches.
- **Hardware & Hyperparameters**: CPU, seed 42, 20 steps, batch size 2, lr $5 \times 10^{-5}$, AdamW, grad clip 1.0.

---

## 2. Quantitative Results

| Metric | Before Training | After Training (Step 20) | Delta | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Train Loss** | 5.4242 | 5.0305 | -0.3937 | Improvement |
| **Train Perplexity** | 226.83 | 153.01 | -73.82 | Improvement |
| **Val Loss (Held-Out)** | 6.9763 | 6.5402 | -0.4361 | Improvement |
| **Val Perplexity (Held-Out)** | 1070.93 | 692.41 | -378.52 | Improvement |
| **Active Tokens Supervised** | — | 5,618 total tokens across 20 steps | — | Normal |
| **Gradient Norm (Mean)** | — | ~4.37 (peak 5.24, final 4.46) | — | Stable |

---

## 3. Qualitative Evaluation & Probe Generations

Deterministic greedy sampling (temperature=0.0, max_new_tokens=40) was evaluated on 5 fixed probes before and after training:

1. **`probe_code_loss`** (*"How is cross entropy loss computed for a single target index in Python?"*):
   - **Before**: `'\n\nStep-by-Step Solution:\nThe total number of the total of the city of ...'` (EOS: False)
   - **After**: `'\n\nStep-by-Step Solution:\nThe total number of the total number of the total number of ...'` (EOS: False)
2. **`probe_code_search`** (*"Explain binary search logic in python."*):
   - **Before**: `'\n\n    >>> linked_list.insert(")\n    >>> linked_list.insert(")\n ...'` (EOS: False)
   - **After**: `'shape[0]\n    >>> linked_list.insert(")\n    >>> linked_list.insert(")\n ...'` (EOS: False)
3. **`probe_rmsnorm`** (*"What is Pre-RMSNorm?"*):
   - **Before**: Repetitive arithmetic template degeneration. (EOS: False)
   - **After**: Repetitive arithmetic template degeneration. (EOS: False)
4. **`probe_general`** (*"What is the capital of India?"*):
   - **Before**: Degenerate sequence. (EOS: False)
   - **After**: Degenerate sequence. (EOS: False)
5. **`probe_val_assert`** (*"How do you check that d_model divides evenly by the number of attention heads in Python?"*):
   - **Before**: Degenerate sequence. (EOS: False)
   - **After**: Degenerate sequence. (EOS: False)

---

## 4. Scientific Findings & Honest Capability Assessment

1. **Loss Dynamics**: Both training loss (-0.39) and held-out validation loss (-0.44) decreased reliably without overfitting or divergence. Optimization behaved normally under AdamW with gradient clipping.
2. **Generative Capability**:
   - **EOS Rate**: 0% both before and after. 20 gradient steps on 39 examples is insufficient to unseat the deep pretraining autoregressive priors of the 3.4M model.
   - **Repetition**: The model remains locked in repetitive pretraining text patterns (`"Step-by-Step Solution:\nThe total number of..."`).
   - **Honest Finding**: While the optimization mechanics succeed and lower perplexity on held-out loss, **no qualitative reasoning or instruction-following capability improvement has been demonstrated**. Capability improvement is declared **INCONCLUSIVE / NOT DEMONSTRATED**.

---

## 5. Artifact & Resumption Verification

- Checkpoint saved atomically to `experiments/batch40_capability_sprint/learning_experiment/sft_checkpoint_step_20.pt`.
- File size: 41,390,763 bytes.
- SHA-256: `fa1edafb821d0a668d27ac30f9bb1bb0c96f2c67b4b0908bda52f08e77c128cc`.
- **Reload Verification**: Successfully reloaded state into a freshly instantiated `ChakrMicro` model and reproduced greedy generation bit-for-bit.
