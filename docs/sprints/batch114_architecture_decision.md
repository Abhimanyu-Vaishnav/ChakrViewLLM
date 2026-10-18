# Batch 114 — Evidence-Based Decision and Handoff

**Sprint**: ChakrView Master Sprint Batches 111–114  
**Module**: Architecture Decision and Sprint Synthesis  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Selected Conclusion & Evidence

### Decision: **Option A**
**(The existing standard autoregressive transformer architecture can learn both variable copying and dynamic argument binding under clean, conversation-aligned sequence supervision).**

### Empirical Evidence Across Batches 111–114:
1. **Verification of Copying Invariance (Batch 111–112)**:
   - Proved that tokenization is strictly compositional across the prompt-target boundary ($\text{encode}(P + T) \equiv \text{encode}(P) + \text{encode}(T)$) across all 100 pairs.
   - Proved that causal shifted-label alignment predicts the first target token directly from the final prompt position without prompt loss leakage.
   - Replicated **100% exact-copy accuracy (20/20)** on the held-out test set under cold reload.
2. **Transfer to Dynamic Tool Arguments (Batch 113)**:
   - Moving from continuous stream chopping to discrete per-sample causal masking completely abolished the constant-call collapse that plagued Batches 95–106.
   - The direct tool-calling model achieved **75.0% (15/20)** exact-pair accuracy on held-out test data.
   - The **curriculum bridge model** (pre-trained on the minimal copy benchmark, then fine-tuned on tool calls) achieved **85.0% (17/20)** exact-pair accuracy on held-out test pairs, **93.8% (15/16)** on validation, and zero constant-call collapse ($5.0\%$).
   - All 17 generated test calls were validly executed by the allowlisted calculator runtime, producing correct arithmetic results.

---

## 2. Answers to the Core Evaluation Questions

### 1. What was actually inspected and changed?
- Audited tokenization compositionality on all 100 pairs, verifying zero boundary discrepancies.
- Verified causal shifted-label indexing in `build_training_sequence()`.
- Replaced the circular regression test in `test_batch107_110_copy_benchmark.py` with evaluator contract tests and 100-pair alignment tests.
- Built a controlled copy-to-tool curriculum dataset (`64 train`, `16 val`, `20 test`, `5 clarification`).
- Added full regression suite in `tests/test_batch111_114_curriculum.py` covering model checkpoints, parser invariants, and allowlisted calculator safety.

### 2. Was the copy benchmark reproduced?
**Yes.** Under cold reload with free-running greedy decoding, the copy task model scored **100.0% (20/20)** on held-out test pairs with 20 unique outputs (collapse rate = 5.0%).

### 3. Did curriculum transfer succeed?
**Yes.** Pre-training on clean variable copying improved tool-argument exact-pair accuracy on held-out test examples from 75.0% (direct control) to **85.0%** (curriculum bridge), with validation exact-pair accuracy reaching **93.8%**.

### 4. Is a pointer/copy mechanism justified?
**No.** The standard 4-layer autoregressive `ChakrMicro` architecture possesses the intrinsic attention capacity to bind dynamic variables to tool arguments when trained on properly bounded sequence windows. Modifying the architecture with a custom pointer network is empirically unnecessary and would introduce complex autograd dependencies and GPU/CPU divergence.

### 5. Known Defects and Unresolved Questions:
- **Clarification False-Positive Rate**: Models trained exclusively on positive tool examples still suffer from 100% false-positive tool calls on non-tool queries.
- **Digit 2 Multi-Token Attractor**: All 3 test errors in the curriculum bridge run involved digit `2` defaulting to digit `1`.

### 6. The Single Most Valuable Next Experiment:
**Negative Calibration & Multi-Token Digit Regularization**:
Train a balanced curriculum that pairs positive tool calls with explicit non-tool clarification examples (50/50 mix) using per-sample masking, while regularizing digit token representations to eliminate the digit-2 neighbor bias.

---

## 3. Checkpoint Manifest

| Condition | Path | SHA-256 |
|---|---|---|
| **Canonical Base** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Stage 1 Copy Model** | `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt` | `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4` |
| **Direct Tool Control** | `experiments/batch111_114_curriculum_transfer/checkpoints/direct_tool_control_step150.pt` | `b4b489fca356b5f12e19eb2b88d7d25a418bb9af543a5f7368f0e13a6f093521` |
| **Curriculum Bridge** | `experiments/batch111_114_curriculum_transfer/checkpoints/curriculum_bridge_transfer_step150.pt` | `67c857666f082aa1057c0c8aef47959f68d661556ba522d4febdcbe5eff8c4a5` |

---

## 4. Reproducible Commands

```powershell
# 1. Run all sprint regression tests (41 tests)
.venv\Scripts\python.exe -m pytest tests/test_batch103_106_slot_weighted_sft.py tests/test_batch107_110_copy_benchmark.py tests/test_batch111_114_curriculum.py -v

# 2. Reproduce the Copy Benchmark cold reload
.venv\Scripts\python.exe -m experiments.batch107_110_operand_copy_benchmark.reproduce_benchmark

# 3. Reproduce Curriculum Transfer experiment
.venv\Scripts\python.exe -m experiments.batch111_114_curriculum_transfer.train_and_eval_curriculum
```
