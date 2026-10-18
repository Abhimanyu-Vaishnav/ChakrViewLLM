# Batch 110 — Architecture Decision and Synthesis

**Sprint**: ChakrView Master Sprint Batches 107–110  
**Module**: Copy Mechanism Justification & Architecture Decision  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Decision Synthesis: Is a Custom Copy Mechanism Justified?

### The Core Empirical Finding
In Batch 109, we proved conclusively that **the standard 4-layer autoregressive transformer architecture (`ChakrMicro`) can learn 100% input-dependent variable copying**:
- **Train exact-pair accuracy**: 64 / 64 (100.0%)
- **Validation exact-pair accuracy**: 16 / 16 (100.0%)
- **Held-out test exact-pair accuracy**: 20 / 20 (100.0%)
- **Constant-output collapse rate**: 5.0% (20 unique outputs across 20 held-out test examples).
- **Target token probability**: >98.6% (Rank 1.0).

### Architecture Decision: **Option C**
**(Investigate training data sequence boundaries, prompt/completion alignment, and curriculum formatting rather than modifying model architecture).**

**Rationale**:
1. **No Architectural Bottleneck**: Standard multi-head causal attention is demonstrably capable of routing prompt tokens across the context window and emitting them at designated output slots. Adding an ad-hoc pointer network, copy head, or hybrid vocabulary readout is empirically unjustified at this stage.
2. **Root Cause of Previous Failure**:
   The failure in Batches 95–106 was caused by:
   - **Stream-level sliding window corruption**: Training on random 64-token slices of a flattened byte stream sliced across conversations, causing the model to see target slots with severed prompt contexts.
   - **JSON boilerplate overshadowing**: The 45-character schema boilerplate (`TOOL_CALL: {"tool": "calculator", "op": "add", "a": ...}`) contains high-frequency static tokens that train early layers to behave as constant n-gram generators.
3. **Hardware & Repository Invariants**: Keeping the canonical architecture unmodified preserves zero-dependency CPU and GPU execution, avoiding custom autograd backward passes or memory-intensive pointer tensors.

---

## 2. Answers to the Four Mandatory Sprint Questions

### 1. Was the previous weighted-loss experiment implemented and evaluated correctly?
- **Yes, with one boundary qualification**: Causal shifted-label indexing, gradient accumulation, optimizer seeding, and evaluation isolation were verified and correct.
- The qualification discovered in Batch 107: the scan-to-quote boundary in `stream_experiment.py` marked trailing JSON punctuation (`}` and `<`) along with the numeric token. However, as demonstrated in both the per-sample script and stream script, loss weighting alone did not teach cross-attention transport across continuous stream slices.

### 2. Did the minimal copy benchmark demonstrate input-dependent generalization?
- **Yes, conclusively.** The model achieved **100.0% exact-pair accuracy** across 20 held-out test pairs containing repeated digits (`(0,0), (4,4), (9,9)`), reverse pairs (`(1,8), (8,1)`), and unseen combinations with zero test-set leakage. Constant-call collapse dropped from 100% to the theoretical minimum of 5.0%.

### 3. Does a pointer/copy mechanism have sufficient empirical justification?
- **No.** Because the baseline transformer architecture achieved 100% generalization on the clean copy benchmark without architectural modifications, the hypothesis that "causal attention cannot bind dynamic variables without a copy head" is refuted.

### 4. What is the single most valuable next experiment?
- **Curriculum Bridge from Minimal Copying to Structured Tool Calling**:
  Fine-tune the model to emit tool calls using a structured sequence boundary objective (per-sample causal masking with prompt exclusion, rather than random stream chopping) and a two-stage curriculum:
  1. *Stage 1*: Copy operands into compact key-value slots (`a={a}, b={b}`).
  2. *Stage 2*: Wrap the dynamic slots in the validated tool execution schema (`TOOL_CALL: {"tool": "calculator", "op": "add", "a": {a}, "b": {b}}`).

---

## 3. Sprint Artifacts & Regression Test Manifest

- `docs/sprints/batch107_experiment_integrity_audit.md`
- `docs/sprints/batch108_operand_copy_benchmark.md`
- `docs/sprints/batch109_copy_learning_results.md`
- `docs/sprints/batch110_architecture_decision.md`
- `tests/test_batch107_110_copy_benchmark.py` (6 passed in 3.73s)
- Total focused suite: **35 passed in 5.51s** (`test_batch103_106_slot_weighted_sft.py` + `test_batch107_110_copy_benchmark.py`).
- Checkpoint: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt` (SHA-256 `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`).
