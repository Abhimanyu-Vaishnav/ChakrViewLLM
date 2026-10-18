# Batch 119 — Checkpoint Lineage & Test-Leakage Audit

**Sprint**: ChakrView Master Sprint Batches 119–122  
**Module**: Test-Leakage Audit, Independent Replication, and Tool-Gating Validation  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Lineage Map of `balanced_curriculum_bridge_step150.pt`

We mapped the full training lineage of the final checkpoint `balanced_curriculum_bridge_step150.pt`:

```text
Canonical Base Checkpoint: step-10,000 (pretraining)
  SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
    │
    ▼
Stage 1 Copy Training: copy_task_sft_step150.pt (150 steps, AdamW, batch=4)
  Data: experiments/batch107_110_operand_copy_benchmark/data/train.jsonl (64 pairs)
  SHA-256: e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4
    │
    ▼
Stage 2 Tool-Gating Curriculum: balanced_curriculum_bridge_step150.pt (150 steps, AdamW, batch=4)
  Data: experiments/batch115_118_tool_gating_curriculum/data/train.jsonl (64 tool additions + 16 non-tool controls)
  SHA-256: ff606e5a8c25272347d1e127005255080eb82192a7b2d6eb8136e9a13134a2f7
```

---

## 2. Detailed Findings on Data Overlap and Split Leakage

We audited the dataset files across Batches 95–118 to answer the required questions with file-level evidence:

### Question 1: Which exact training records produced the copy-task checkpoint?
`copy_task_sft_step150.pt` was trained on `experiments/batch107_110_operand_copy_benchmark/data/train.jsonl` (64 records).

### Question 2: Were any of the 20 Batch 115–118 test pairs present in its training data?
**Zero.**
- The 20 held-out test pairs in Batch 108 were defined as:
  `[(0,0), (9,9), (4,4), (1,8), (8,1), (2,7), (7,2), (3,6), (6,3), (0,5), (5,0), (1,4), (4,1), (2,9), (9,2), (3,5), (5,3), (6,8), (8,6), (7,9)]`.
- In `generate_copy_data.py`, these 20 pairs were explicitly withheld before generating `train.jsonl`.
- Verification check: `set(train_pairs) & set(test_pairs) == set()` (0 pairs overlap).

### Question 3: Were any test prompts or targets included in prior historical training (e.g., Batch 95–98)?
**Audit Finding**:
- While the Stage 1 copy training (`train.jsonl`) had zero overlap with the 20 test pairs, the earlier Batch 95 continuous token stream (`train_slot_tokens.bin`) was generated from `experiments/batch95_98_slot_binding_diagnostics/data/train.jsonl`.
- Of the 20 test pairs used in Batches 108–118, **11 pairs** had appeared in that historical Batch 95 training file:
  `[(0,0), (0,5), (2,7), (3,5), (5,0), (5,3), (6,8), (7,2), (7,9), (8,6), (9,9)]`.
- However, **9 pairs** were completely absent from all training stages in the entire repository history:
  `[(1,4), (1,8), (2,9), (3,6), (4,1), (4,4), (6,3), (8,1), (9,2)]`.

### Question 4: Was the test set used to choose the curriculum or architecture?
- Validation sets ($N=16$ tool, $N=4$ non-tool) were tracked during development.
- The held-out test set ($N=20$ tool, $N=8$ non-tool) was evaluated only as a cold-reload evaluation in Batch 117 without tuning or threshold filtering.

### Question 5: Does the reported 100% accuracy survive on the strictly unseen pairs?
- Inspecting the per-example predictions from Batch 117 specifically for the 9 pairs that were never seen in any training stage in repository history:
  - $(1, 4) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 1, \text{"b": } 4\} \to \text{Correct: True}$
  - $(1, 8) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 1, \text{"b": } 8\} \to \text{Correct: True}$
  - $(2, 9) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 2, \text{"b": } 9\} \to \text{Correct: True}$
  - $(3, 6) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 3, \text{"b": } 6\} \to \text{Correct: True}$
  - $(4, 1) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 4, \text{"b": } 1\} \to \text{Correct: True}$
  - $(4, 4) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 4, \text{"b": } 4\} \to \text{Correct: True}$
  - $(6, 3) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 6, \text{"b": } 3\} \to \text{Correct: True}$
  - $(8, 1) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 8, \text{"b": } 1\} \to \text{Correct: True}$
  - $(9, 2) \to \text{Generated: } \text{TOOL\_CALL: } \{..., \text{"a": } 9, \text{"b": } 2\} \to \text{Correct: True}$
- **Result**: Accuracy on the 9 strictly unseen pairs was **100.0% (9/9)**.

---

## 3. Leakage Status Summary

1. **Within-Experiment Leakage**: None. Zero prompt or target overlap between `train.jsonl` and `test.jsonl` in Batches 108–118.
2. **Lineage Cross-Stage Overlap**: 11 out of 20 test pairs appeared in historical Batch 95 token streams, while 9 pairs were never trained in any prior experiment.
3. **Independent Replication Mandate**: To eliminate all ambiguity regarding historical lineage overlap, Batch 121 will evaluate an independent benchmark comprising arithmetic pairs absent from all prior training data wherever possible, along with fresh prompt phrasings.
