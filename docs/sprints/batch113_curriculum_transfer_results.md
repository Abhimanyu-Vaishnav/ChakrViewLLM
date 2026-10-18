# Batch 113 — Conditional Controlled Curriculum Experiment

**Sprint**: ChakrView Master Sprint Batches 111–114  
**Module**: Copy-to-Tool Curriculum Transfer  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Experimental Design & Hypotheses

Following the validated reproduction of the Operand Copy Benchmark in Batch 112, we tested whether learning input-to-output variable routing transfers to dynamic tool-argument binding in the secure calculator runtime.

### Hypotheses:
1. **Hypothesis 1 (Per-Sample Formatting Hypothesis)**: Training on discrete, conversation-aligned sequence windows (where the prompt is causally masked with `-100` and the completion begins with a clear prompt boundary) resolves constant-call collapse without requiring stream sliding-window chopping.
2. **Hypothesis 2 (Curriculum Bridge Hypothesis)**: Initializing from a model that has already learned clean variable copying (Stage 1 copy checkpoint) will accelerate argument binding and achieve higher held-out exact-pair accuracy than direct SFT from the canonical base checkpoint.

---

## 2. Controlled Conditions & Token Budget

Both conditions were trained on the identical 64-example calculator tool training set with identical hyperparameters:
- **Optimizer**: AdamW ($\text{lr} = 3\times 10^{-4}$, $\text{weight\_decay} = 0.01$, gradient norm clip = 1.0).
- **Training Budget**: 150 optimizer steps, batch size = 4 (600 sequence exposures total).
- **Seed**: 42.
- **Arms**:
  - **Arm 1 (Direct Tool Control)**: Started from the canonical step-10,000 checkpoint (`training_checkpoint_step_10000.pt`, SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
  - **Arm 2 (Curriculum Bridge Transfer)**: Started from the copy-task checkpoint (`copy_task_sft_step150.pt`, SHA-256 `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`).

---

## 3. Results Across Splits (Cold Reload Evaluation)

Both models were evaluated on the held-out splits under free-running greedy decoding:

| Condition | Split | $N$ | Valid Syntax | Tool / Op Acc | Op A Acc | Op B Acc | Exact Pair Acc | Exec Acc | Unique Outputs | Collapse Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **Direct Tool Control** | Train | 64 | 100.0% | 100.0% | 81.2% | 93.8% | **75.0%** | 75.0% | 53 | 3.1% |
| *(Canonical Base -> SFT)* | Val | 16 | 100.0% | 100.0% | 75.0% | 68.8% | **50.0%** | 50.0% | 16 | 6.2% |
| | Test | 20 | 100.0% | 100.0% | 80.0% | 90.0% | **75.0%** | 75.0% | 19 | 10.0% |
| **Curriculum Bridge** | Train | 64 | 100.0% | 100.0% | 90.6% | 95.3% | **85.9%** | 85.9% | 59 | 4.7% |
| *(Copy Task -> SFT)* | Val | 16 | 100.0% | 100.0% | 93.8% | 100.0% | **93.8%** | 93.8% | 16 | 6.2% |
| | Test | 20 | 100.0% | 100.0% | 90.0% | 95.0% | **85.0%** | 85.0% | 20 | 5.0% |

---

## 4. Generalization Analysis & Failure Modes

1. **Breakthrough in Constant-Call Collapse**:
   - In Batches 95–106, continuous stream SFT resulted in $100\%$ constant-call collapse (`{"a": 1, "b": 8}` or `{"a": 3, "b": 8}`).
   - Under per-sample sequence masking, **constant-call collapse dropped to $5.0\%$** (20 unique calls across 20 held-out test examples).
2. **Transfer Efficiency**:
   - The Curriculum Bridge model achieved **85.0% (17/20)** exact argument pair accuracy on the held-out test set, and **93.8% (15/16)** on validation.
   - Every correctly bound tool call was validated and safely executed through the allowlisted calculator runtime, producing 100% mathematical correctness on those calls.
3. **Failure Analysis on Remaining 3 Test Cases**:
   - In the 3 failure cases on the test set:
     - Expected `(2, 7)` $\to$ Generated `{"a": 1, "b": 7}`
     - Expected `(2, 9)` $\to$ Generated `{"a": 1, "b": 9}`
     - Expected `(9, 2)` $\to$ Generated `{"a": 9, "b": 1}`
   - All 3 errors involved digit `2` defaulting to digit `1` in one slot. Notice that digit `2` is a multi-token neighbor in `vocab_4096`.
4. **Clarification Control Security**:
   - Both models exhibited $100\%$ (5/5) false-positive tool calls on non-tool queries when trained on positive examples only. The allowlisted calculator runtime successfully rejected all invalid tools (status `UNSUPPORTED_TOOL`).

---

## 5. Artifact Hashes

- **Direct Tool Control Checkpoint**:
  - `experiments/batch111_114_curriculum_transfer/checkpoints/direct_tool_control_step150.pt`
  - SHA-256: `b4b489fca356b5f12e19eb2b88d7d25a418bb9af543a5f7368f0e13a6f093521`
- **Curriculum Bridge Checkpoint**:
  - `experiments/batch111_114_curriculum_transfer/checkpoints/curriculum_bridge_transfer_step150.pt`
  - SHA-256: `67c857666f082aa1057c0c8aef47959f68d661556ba522d4febdcbe5eff8c4a5`
