# Batch 111 — Copy Benchmark Integrity and Tokenization Audit

**Sprint**: ChakrView Master Sprint Batches 111–114  
**Module**: Copy-Benchmark Integrity, Reproducibility, and Conditional Tool-Argument Curriculum  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Repository State and Checkpoint Verification

- **Branch**: `dev/v0.6-v0.7-validation`
- **Current HEAD**: `b4e0189a9f1ccf4466137db11d9647ec8d473d04`
- **Working Tree**: User modification in `chakrview/runtime/session.py` and existing workbench reports strictly preserved without staging.
- **Canonical Step-10,000 Checkpoint**:
  - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Size: 42,212,787 bytes
  - Calculated SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Status: Bit-exact verification confirmed.
- **Reported Batch 109 Checkpoint**:
  - File: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt`
  - Size: 41,388,179 bytes
  - Calculated SHA-256: `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`

---

## 2. Tokenization Boundary Compositionality Audit

We tested the compositionality assumption:
$$\text{encode}(\text{prompt} + \text{target}) \stackrel{?}{=} \text{encode}(\text{prompt}) + \text{encode}(\text{target})$$
across all 100 possible operand pairs $(a, b) \in \{0..9\}^2$.

### Results:
- **Total pairs tested**: 100 / 100.
- **Mismatches found**: **0**.
- **First differing token**: None.
- **Verification Details**:
  - Every prompt ends with `<|im_start|>assistant\n` (last token: `\n`, ID: `13`).
  - Every target begins with `A=` (first token: `A`, ID: `68`).
  - Because `\n` and `A` do not form a merged BPE merge pair in `vocab_4096`, the tokenization across the boundary is strictly compositional:
    $$\text{encode}(\text{prompt} + \text{target}) \equiv \text{encode}(\text{prompt}) \mathbin{\Vert} \text{encode}(\text{target})$$
  - Prompt length is **strictly 43 tokens** for all 100 pairs.
  - Target length is **strictly 16 tokens** for all 100 pairs.

---

## 3. Shifted-Label Training Sequence Audit

We audited `build_training_sequence()` in `train_copy_task.py`:
1. **Target Masking**:
   - `prompt_tokens` length = 43.
   - `target_ids = [-100] * (43 + 1) + target_tokens + [1]`.
   - Length of `target_ids` equals length of `input_ids` ($61$ tokens).
2. **Shifted Alignment**:
   - In autoregressive modeling, `x_in = input_ids[:-1]` (length 60), `y_tgt = target_ids[1:]` (length 60).
   - At sequence position 43 (0-indexed):
     - `x_in[43]` is the 43rd token of `input_ids`, which is the final prompt token: `\n` (13).
     - `y_tgt[43]` is `target_ids[44]`, which is `target_tokens[0]`: `A` (68).
     - **Result**: The first target token is predicted directly from the final prompt position.
3. **Loss Exclusion**:
   - All positions `0 .. 42` in `y_tgt` have target value `-100`.
   - Zero training loss is backpropagated through prompt tokens.
   - Target tokens receive loss exactly once.
   - EOS token (`1`) is supervised exactly once at the sequence termination.

---

## 4. Evaluator Correctness Audit

1. **Prompt Isolation**:
   - In `evaluate_copy_model()`, greedy generation starts strictly from `[0] + tok.encode(record['prompt'])`.
   - The generation prefix contains zero completion tokens or answer labels.
2. **Output Parsing**:
   - `parse_copy_output()` requires explicit `A=\d+` and `B=\d+` patterns.
   - Empty, truncated, or malformed outputs parse to `None` and score 0.0 accuracy.
3. **Metric Distinctions**:
   - Primary benchmark metric is **free-running greedy generation accuracy**, requiring the model to generate both operands correctly in a single turn without external hints.
   - Teacher-forced probabilities ($P(A)$ and $P(B)$) are strictly separated and designated as internal diagnostic probes.
4. **Circular Test Removal**:
   - Removed circular test assertion enforcing 20/20 in `test_batch107_110_copy_benchmark.py`.
   - Replaced with structural evaluator contract tests and 100-pair alignment tests.
   - All 7 tests in `tests/test_batch107_110_copy_benchmark.py` pass cleanly.
