# Batch 106 — Synthesis, Regression Tests, and Handoff

**Sprint**: ChakrView Master Sprint Batches 103–106  
**Module**: Dynamic Argument Learning: Slot-Weighted SFT and Generalization Audit  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Executive Summary & Verification

Across Batches 103–106, we executed a rigorous, controlled experiment evaluating whether upweighting loss supervision on argument value tokens ($5.0\times$ slot weight) resolves constant-output collapse and induces dynamic argument binding in ChakrMicro.

### What Was Verified:
1. **Canonical Checkpoint Immutability**:
   - `training_checkpoint_step_10000.pt` verified with bit-exact SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
2. **Causal Control Equivalence**:
   - Condition B (Uniform Control) and Condition C (Slot-Weighted 5×) started from the identical canonical checkpoint, trained on the identical 71,139-token data stream with identical random seeds, batching schedules, optimizer parameters (AdamW, lr=5e-4, 100 steps), and token budgets (25,600 token exposures).
3. **Loss-Masking Precision**:
   - Unit tests confirmed that exactly the intended argument positions received $5.0\times$ weight, handling multi-token values and preserving proper next-token target alignment without prompt target leakage.
4. **Comprehensive Test Suite**:
   - 29/29 dedicated regression tests in `tests/test_batch103_106_slot_weighted_sft.py` passed in 3.54s.
   - Full repository test suite passed completely (`2583 passed, 3 skipped in 811.03s`).

---

## 2. Synthesis of Findings

### 1. What Changed Relative to the Uniform-Loss Control?
- **Teacher-forced target probabilities** increased slightly:
  - Operand A: from $8.71\%$ (Rank 5.9) to $8.88\%$ (Rank 5.8).
  - Operand B: from $9.83\%$ (Rank 4.7) to $10.95\%$ (Rank 4.8).
- **The modal output token shifted**:
  - Uniform Control collapsed to `{"a": 1, "b": 8}` (100% of samples).
  - Slot-Weighted 5× collapsed predominantly to `{"a": 3, "b": 8}` (77.8%–88.9% of samples).

### 2. Did Greedy Argument Generation Actually Improve?
**No.** While the extra loss penalty moved the model away from the `1, 8` basin, it landed directly into a neighboring static attractor basin (`3, 8`). The model's greedy output remains unconditioned on the input prompt operands.

### 3. Did Held-Out Exact-Pair Accuracy Improve?
**No.** Held-out test exact-pair accuracy was **0.0% (0/18)** for Condition C, compared to 5.6% (1/18) for Condition B. The single hit in Condition B was an artifact of the static prediction `(1, 8)` coinciding with one test example. Neither model possesses input-conditioned argument generation.

### 4. Do Constant-Call Collapse and Spurious Tool Calls Remain?
**Yes.**
- Constant-call collapse remains severe ($\ge 77.8\%$).
- Spurious tool calls remain at 100% (5/5) across out-of-domain requests, confirming that standard next-token SFT without negative calibration destroys the tool decision gate.

### 5. What Remains Inconclusive?
Whether larger scale loss weighting (e.g., $20\times$) or curriculum warmups could force attention heads to route tokens, or whether standard causal attention without copy mechanisms/pointer attention is fundamentally incapable of zero-shot numeric binding with small parameters (1.5M).

### 6. The Single Most Valuable Next Experiment
**Pointer-Augmented Attention / Copy Head Supervision**: Rather than merely increasing standard vocabulary cross-entropy loss, introduce an explicit attention guide or pointer loss that trains the attention matrix at the generation slot to place high attention weight specifically on the prompt operand tokens.

---

## 3. Checkpoint Manifest & Reproducibility Hashes

| Condition | File Path | Steps | SHA-256 |
|---|---|---|---|
| **Canonical Base** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 10,000 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Condition A (Cond C Base)** | `experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt` | 100 | `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2` |
| **Condition B (Uniform Ctrl)** | `experiments/batch103_106_slot_weighted_sft/checkpoints/uniform_loss_control_stream_step100.pt` | 100 | `7967ab078705c60643f795878c06539c349b0efee728cc04c9699b39d21bf5e5` |
| **Condition C (Slot-Weighted 5×)** | `experiments/batch103_106_slot_weighted_sft/checkpoints/slot_weighted_5x_stream_step100.pt` | 100 | `7b057042dc375444406cd02de552fc72ff98257cc92e92046a2bc45846946e7a` |

---

## 4. Reproducible Commands

```powershell
# 1. Run regression test suite
.venv\Scripts\python.exe -m pytest tests/test_batch103_106_slot_weighted_sft.py -v

# 2. Run full test suite
.venv\Scripts\python.exe -m pytest tests/ -q

# 3. Re-run controlled experiment
.venv\Scripts\python.exe -m experiments.batch103_106_slot_weighted_sft.stream_experiment

# 4. Measure teacher-forced probabilities
.venv\Scripts\python.exe -m experiments.batch103_106_slot_weighted_sft.eval_teacher_probs
```
