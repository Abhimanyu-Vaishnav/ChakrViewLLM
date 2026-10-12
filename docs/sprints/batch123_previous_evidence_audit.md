# Batch 123 — Audit of Previous Evidence & Benchmark Contract

## Executive Summary

Batch 123 audited the previous experimental evidence from Batches 115–122 across:
- `experiments/batch119_122_independent_replication/`
- `experiments/batch115_118_tool_gating_curriculum/`
- `experiments/batch107_110_operand_copy_benchmark/`
- `experiments/batch95_98_slot_binding_diagnostics/`

This audit verifies all claims, checks operand and reversed-pair overlap, inspects prompt-template leakage, identifies token stream limitations, and validates the reported 15/15 canonical-template and 2/10 fresh-phrasing results with exact file-level evidence and confirmed denominators.

---

## 1. Lineage Audit & Checkpoint Verification

### Verified Artifacts & Checkpoints
1. **Canonical ChakrMicro v0.1 Checkpoint**:
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Size: 42,212,787 bytes
   - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-exact).
2. **Stage 1 Copy-Task Pre-Trained Checkpoint**:
   - Path: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt`
   - SHA-256: `e84b2ea9b633a277977eb7b686d1ff8b8e054457e5d8b2d1ef9569fa1e9753e8`
   - Produced by 150 steps of AdamW ($lr=3\times 10^{-4}$) on 64 training pairs in `copy_task_sft`.
3. **Curriculum Bridge Checkpoint**:
   - Path: `experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_curriculum_bridge_step150.pt`
   - SHA-256: `ff606e5a8c25272347d1e127005255080eb82192a7b2d6eb8136e9a13134a2f7`
   - Initialized from `copy_task_sft_step150.pt`, trained on 80 records (64 tool additions, 16 non-tool controls).
4. **Canonical Control Checkpoint**:
   - Path: `experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_control_base_step150.pt`
   - SHA-256: `6905fa8b759f3730926665395e54453a779fb0767fb574e115163b3fc512a427`
   - Initialized directly from canonical `training_checkpoint_step_10000.pt`, trained on identical 80 curriculum records.

---

## 2. Comprehensive Overlap & Leakage Audit

### A. Operand-Pair & Reversed-Pair Analysis
Across the entire repository training history:
- Total possible single-digit pairs: 100 ($0..9 \times 0..9$).
- Unique ordered training pairs present across historical training files:
  - Batch 95 (`train.jsonl`): 80 pairs
  - Batch 108 (`train.jsonl`): 64 pairs
  - Batch 116 (`train.jsonl`): 64 pairs
  - **Union of all trained ordered pairs**: **85 pairs**.
- **15 single-digit pairs were never trained in repository history**:
  `[(1, 4), (1, 8), (2, 8), (2, 9), (3, 6), (4, 1), (4, 4), (4, 8), (4, 9), (6, 3), (7, 4), (7, 7), (8, 1), (9, 0), (9, 2)]`.

### B. Reversed-Pair (Symmetry) Audit on Batch 121 Test Pairs
When examining whether the reverse ordered pair $(b, a)$ appeared in any prior training split:
- **5 pairs** had their reverse in prior training:
  `{(7, 4), (9, 0), (4, 9), (4, 8), (2, 8)}` (because $(4, 7), (0, 9), (9, 4), (8, 4), (8, 2)$ were in training).
- **10 pairs** were **symmetrically unseen** (neither $(a, b)$ nor $(b, a)$ ever occurred in any training set):
  `[(1, 4), (1, 8), (2, 9), (3, 6), (4, 1), (4, 4), (6, 3), (7, 7), (8, 1), (9, 2)]`.

### C. Performance on Symmetrically Unseen Pairs (Canonical Template)
Inspecting the raw prediction records from `experiments/batch119_122_independent_replication/reports/replication_results.json`:
- **Reversed-pair seen (N=5)**: Exact call = **5/5 (100.0%)**
- **Symmetrically unseen (N=10)**: Exact call = **10/10 (100.0%)**
  - E.g.: `(1, 4)` -> `{"a": 1, "b": 4}` (Exact: True)
  - E.g.: `(1, 8)` -> `{"a": 1, "b": 8}` (Exact: True)
  - E.g.: `(2, 9)` -> `{"a": 2, "b": 9}` (Exact: True)
  - E.g.: `(3, 6)` -> `{"a": 3, "b": 6}` (Exact: True)
  - E.g.: `(4, 1)` -> `{"a": 4, "b": 1}` (Exact: True)
  - E.g.: `(4, 4)` -> `{"a": 4, "b": 4}` (Exact: True)
  - E.g.: `(6, 3)` -> `{"a": 6, "b": 3}` (Exact: True)
  - E.g.: `(7, 7)` -> `{"a": 7, "b": 7}` (Exact: True)
  - E.g.: `(8, 1)` -> `{"a": 8, "b": 1}` (Exact: True)
  - E.g.: `(9, 2)` -> `{"a": 9, "b": 2}` (Exact: True)

**Confirmation**: The 15/15 result is verified with exact per-example predictions and survives both ordered and reversed pair leakage audits.

---

## 3. Fresh Phrasings Failure Breakdown (2/10 Verified)

In Batch 121, 10 examples evaluated the curriculum bridge model on prompt phrasings never seen during training:
- **Exact-Call Accuracy**: **2/10 (20.0%)**
  - `REPL-FRESH-PHRASING-06` (`"Calculate total: A=4 + B=4."`) -> exact call `{"a": 4, "b": 4}` (True)
  - `REPL-FRESH-PHRASING-09` (`"Add together A=9 and B=2."`) -> exact call `{"a": 9, "b": 2}` (True)
- **Error Pattern in the Remaining 8 Examples**:
  - Operand B accuracy was **8/10 (80.0%)**: the model successfully parsed and copied the second operand `b` from the prompt text.
  - Operand A accuracy was **2/10 (20.0%)**: for the first operand, the model collapsed to emitting `a: 2` in 7 out of 8 failed cases.
  - Cause: In the training set, 100% of tool additions followed the exact multi-line template:
    `Calculate sum of:\nA={a}\nB={b}`.
    When flattened into natural sentence structures (e.g., `"Please add A=1 and B=4."`), the positional attention offset shifted, breaking operand A extraction while leaving operand B (which immediately precedes the end-of-prompt boundary) relatively intact.

---

## 4. Historical Token Streams Audit Limitations

- In `experiments/batch95_98_slot_binding_diagnostics/data/`, the binary token stream `train_slot_tokens.bin` was verified to be a token-packed representation of `train.jsonl` ($N=140$ records: 80 tool, 60 controls).
- Earlier pretraining data for the base canonical checkpoint (steps 1–10,000) from `experiments/batch16_21_combined/` consists of broad general corpus text. While single-digit digits appeared throughout English text during pretraining, structured tool-call formats (`TOOL_CALL: {"tool": "calculator"...}`) were completely absent until the SFT batches.
- **Unresolved Limitation**: Because general pretraining corpus tokens include natural English numbers, zero-shot digit knowledge exists in the base model embeddings, but tool-conditioned binding is solely acquired during SFT.

---

## 5. Contract for the Two-Digit Benchmark (Batches 124–126)

To build a definitive test of tool-conditioned argument binding:
1. **Operand Domain**: Two-digit integers ($10..99$).
2. **Template Diversity**: At least 8 distinct training prompt templates, with at least 2 templates reserved exclusively for held-out evaluation.
3. **Isolations**:
   - Familiar operand pairs with held-out prompt templates.
   - Unseen ordered operand pairs with familiar templates.
   - Unseen operand pairs with held-out templates.
   - Negative controls & underspecified queries.
4. **Execution Protocol**: Cold-reloaded checkpoints, free-running greedy decoding, zero hardcoded tokens, allowlisted runtime verification.
