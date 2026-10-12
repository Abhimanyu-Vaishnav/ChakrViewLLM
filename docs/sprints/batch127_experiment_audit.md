# Batch 127 — Independent Evidence Audit of Batches 123–126

## Executive Summary

Batch 127 performed an independent, source-level evidence audit of the Batch 123–126 experimental suite (`experiments/batch123_126_prompt_diverse_multidigit/`), inspecting dataset generation, split isolation, checkpoint lineage, metrics computation, and evaluation parser semantics.

---

## 1. Lineage & Checkpoint Hash Audit

### Verified Artifacts
1. **Canonical ChakrMicro v0.1 Checkpoint**:
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Bit-exact match).
2. **Stage 1 Copy-Task Initialization Checkpoint**:
   - Path: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt`
   - Computed SHA-256: `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`
   - *Audit Discrepancy Found*: The report in `batch123_previous_evidence_audit.md` reported `e84b2ea9b633a277977eb7b686d1ff8b8e054457e5d8b2d1ef9569fa1e9753e8`, whereas the actual checkpoint on disk has hash `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4` (which matches Batches 109, 110, 111, 112, 114, 118, and 119). The discrepancy was a typographical error in the Batch 123 report text.

---

## 2. Dataset Overlap & Leakage Audit

### Dataset Verification (`manifest.json`):
- Train (`train.jsonl`, SHA-256: `a41bf147...`): 100 records (70 tool additions across 60 unique two-digit pairs; 30 negative controls).
- Validation (`val.jsonl`, SHA-256: `aa2b6935...`): 20 records (15 tool, 5 negative controls).
- Test (`test.jsonl`, SHA-256: `d1efbf1d...`): 45 records (35 tool, 10 negative controls).

### Leakage Verification Results:
- **Exact Prompt Overlap**:
  - `train` vs `val`: 5 prompts overlapped (intentional validation check).
  - `train` vs `test`: **0 prompts overlapped**.
  - `val` vs `test`: **0 prompts overlapped**.
- **Operand Pair Overlap in Test**:
  - Partition 1 (`familiar_pairs_unseen_templates`, $N=10$): 10/10 pairs present in training pool (intended familiar test).
  - Partition 2 (`unseen_pairs_familiar_templates`, $N=15$): **0/15 pairs in train**, **0/15 reversed pairs in train**.
  - Partition 3 (`unseen_pairs_unseen_templates`, $N=10$): **0/10 pairs in train**, **0/10 reversed pairs in train**.
  - Partition 4 (`test_negative_control`, $N=10$): 10 fresh out-of-domain and underspecified prompts, **0 overlap with train controls**.

---

## 3. Metrics Recomputation & Independent Verification

Recomputation directly from saved predictions in `multidigit_experiment_results.json`:

| Metric | Stated Arm A | Verified Arm A | Stated Arm B | Verified Arm B | Match? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tool Examples ($N$)** | 35 | 35 | 35 | 35 | Verified |
| **Parse Rate** | 71.4% (25/35) | 25/35 (71.43%) | 71.4% (25/35) | 25/35 (71.43%) | Verified |
| **Schema Validity Rate** | 71.4% (25/35) | 25/35 (71.43%) | 71.4% (25/35) | 25/35 (71.43%) | Verified |
| **Operand A Accuracy** | 20.0% (7/35) | 7/35 (20.00%) | 34.3% (12/35) | 12/35 (34.29%) | Verified |
| **Operand B Accuracy** | 11.4% (4/35) | 4/35 (11.43%) | 28.6% (10/35) | 10/35 (28.57%) | Verified |
| **Exact-Call Accuracy** | 5.7% (2/35) | 2/35 (5.71%) | 20.0% (7/35) | 7/35 (20.00%) | Verified |
| **Swapped Arguments ($a \leftrightarrow b$)** | 0 | 0/25 (0.00%) | 0 | 0/25 (0.00%) | Verified |
| **Negative Controls ($N$)** | 10 | 10 | 10 | 10 | Verified |
| **False-Positive Rate** | 0.0% (0/10) | 0/10 (0.00%) | 0.0% (0/10) | 0/10 (0.00%) | Verified |
| **Clean Refusal Rate** | 100.0% (10/10) | 10/10 (100.00%) | 100.0% (10/10) | 10/10 (100.00%) | Verified |

---

## 4. Evidence Table

| Claim | Evidence Inspected | Independently Verified? | Limitation | Consequence for Next Experiment |
| :--- | :--- | :---: | :--- | :--- |
| Zero train-test leakage for unseen pairs | Evaluated all 25 unseen pairs against all 60 train pairs and reversed pairs | **Yes** | Individual operand numbers appeared in other train pairs | Must measure both exact-pair and individual operand generalization |
| Arm B outperforms Arm A on exact-call accuracy | Cold-reload evaluation predictions and metrics | **Yes** | Overall exact-call accuracy is only 20% | Copy-task initialization helps, but is insufficient for robust multi-digit binding |
| Model preserves operand order without swapping | Inspected all 25 parsed tool calls in both arms | **Yes** | Exact-match on both operands remains low | Order preserving attention is intact; binding accuracy is the true failure mode |
| 100% clean refusal on negative controls | Inspected all 10 non-tool predictions in both arms | **Yes** | None | Tool gating remains 100% robust across both arms |
| BPE segmentation causes 2-digit degradation | Tokenized all 90 two-digit numbers and checked error distribution | **Partial** | Comma-merger occurs across 70% of numbers, but accuracy on merged vs unmerged is comparable | BPE is a contributing factor, but finite pair exposure during training is a major confound |
