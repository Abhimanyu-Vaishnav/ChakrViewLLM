# Batch 109 — Copy Task Learning Results

**Sprint**: ChakrView Master Sprint Batches 107–110  
**Module**: Controlled Copy Learning Experiment  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Experimental Setup

We trained a single, bounded copy experiment starting directly from the canonical step-10,000 checkpoint:
- **Base Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Data**: 64 training records from the Operand Copy Benchmark (`train.jsonl`).
- **Loss Masking**: Autoregressive causal Cross-Entropy with `ignore_index=-100` on prompt tokens (loss computed strictly on completion).
- **Optimizer**: AdamW ($\text{lr} = 3\times 10^{-4}$, $\text{weight\_decay} = 0.01$, gradient clip = 1.0).
- **Budget**: 150 optimizer steps, batch size = 4 (600 sequence exposures total).
- **Seed**: 42.

---

## 2. Training Curve & Checkpoint Immutability

| Step | Loss |
|---|---|
| Step 1 | 6.1220 |
| Step 25 | 0.2747 |
| Step 50 | 0.2916 |
| Step 75 | 0.2638 |
| Step 100 | 0.0949 |
| Step 125 | 0.0047 |
| Step 150 | 0.0022 |

- **Saved Checkpoint**: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt`
- **SHA-256**: `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`

---

## 3. Generalization & Test Results

The saved checkpoint was reloaded into a fresh `ChakrMicro` model instance and evaluated under greedy decoding (`max_new_tokens=25`):

| Split | $N$ | Operand A Acc | Operand B Acc | Exact Pair Acc | Unique Outputs | Collapse Rate | Mean P(A) [Rank] | Mean P(B) [Rank] |
|---|---|---|---|---|---|---|---|---|
| **Train** | 64 | **100.0%** | **100.0%** | **100.0%** | 64 / 64 | 1.56% | 98.75% [1.0] | 98.71% [1.0] |
| **Val** | 16 | **100.0%** | **100.0%** | **100.0%** | 16 / 16 | 6.25% | 98.61% [1.0] | 98.47% [1.0] |
| **Test** | 20 | **100.0%** | **100.0%** | **100.0%** | 20 / 20 | 5.00% | 98.72% [1.0] | 98.61% [1.0] |

---

## 4. Analysis of Held-Out Generalization

All 20 held-out test pairs—which had **zero overlap** with the training set—were reproduced with 100% exact-match accuracy under greedy decoding:

```text
  Input: [0, 0] -> Gen: 'A=0\nB=0' -> Match: True (repeated digit)
  Input: [9, 9] -> Gen: 'A=9\nB=9' -> Match: True (repeated digit)
  Input: [4, 4] -> Gen: 'A=4\nB=4' -> Match: True (repeated digit)
  Input: [1, 8] -> Gen: 'A=1\nB=8' -> Match: True (reverse pair)
  Input: [8, 1] -> Gen: 'A=8\nB=1' -> Match: True (reverse pair)
  Input: [2, 7] -> Gen: 'A=2\nB=7' -> Match: True (reverse pair)
  Input: [7, 2] -> Gen: 'A=7\nB=2' -> Match: True (reverse pair)
  Input: [3, 6] -> Gen: 'A=3\nB=6' -> Match: True (reverse pair)
  Input: [6, 3] -> Gen: 'A=6\nB=3' -> Match: True (reverse pair)
  Input: [0, 5] -> Gen: 'A=0\nB=5' -> Match: True (reverse pair with 0)
  Input: [5, 0] -> Gen: 'A=5\nB=0' -> Match: True (reverse pair with 0)
  Input: [1, 4] -> Gen: 'A=1\nB=4' -> Match: True (reverse pair)
  Input: [4, 1] -> Gen: 'A=4\nB=1' -> Match: True (reverse pair)
  Input: [2, 9] -> Gen: 'A=2\nB=9' -> Match: True (reverse pair with 9)
  Input: [9, 2] -> Gen: 'A=9\nB=2' -> Match: True (reverse pair with 9)
  Input: [3, 5] -> Gen: 'A=3\nB=5' -> Match: True (reverse pair)
  Input: [5, 3] -> Gen: 'A=5\nB=3' -> Match: True (reverse pair)
  Input: [6, 8] -> Gen: 'A=6\nB=8' -> Match: True (reverse pair)
  Input: [8, 6] -> Gen: 'A=8\nB=6' -> Match: True (reverse pair)
  Input: [7, 9] -> Gen: 'A=7\nB=9' -> Match: True (asymmetric pair)
```

### Key Mechanistic Insights:
1. **Transformer Attention is Fully Capable of Dynamic Variable Routing**:
   The standard 4-layer autoregressive `ChakrMicro` architecture possesses sufficient attention capacity to route input operands to generation positions when the output format is clean and uncorrupted by competing attractors.
2. **Zero Constant-Call Collapse**:
   Across all 20 test examples, 20 unique outputs were generated (collapse rate = 5.0%, representing the theoretical minimum of $1/20$).
3. **Teacher-Forced Confidence**:
   Target probabilities for both Operand A and Operand B exceeded $98.6\%$ (Rank 1.0 across all samples).
4. **Diagnosis of Previous Failures**:
   The failure in Batches 95–106 was **not an architectural deficiency** in the transformer's capacity to copy. The failure was caused by:
   - SFT on continuous Wikipedia token streams where the model was trained with random sequence offsets ($64$ tokens) rather than aligned sequence boundaries.
   - Long JSON formatting boilerplate (`TOOL_CALL: {"tool": "calculator", "op": "add", ...}`) where early layers attend strongly to constant static schema tokens, overwhelming the weak cross-attention signal to the prompt operands.
