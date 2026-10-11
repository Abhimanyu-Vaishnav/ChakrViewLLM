# Code Review: Accumulated Work (Batches 28–39) & Production Readiness

**Audit Date:** 2026-10-11  
**Working Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `7c3be923acf997d98f693b3509bece86ddf97641`  

---

## 1. Executive Summary

Since commit `7c3be92` (Batch 32), the repository has operated under an absolute Git preservation policy. Seven comprehensive validation sprints (Batches 33 through 39) have generated significant research tools, diagnostics, and test suites that currently exist as untracked files on disk. In addition, one tracked production file—`chakrview/runtime/session.py`—contains an active user/batch modification.

This review classifies all accumulated components across Batches 28–39, analyzes technical integrity, assesses integration status, and identifies key architectural bottlenecks.

---

## 2. Classification of Accumulated Work

Each examined component is classified according to the governance schema:
- **Stable implementation candidate**: Production code ready for integration into `chakrview/`.
- **Experimental artifact**: Standalone batch scripts, exploratory harnesses, or interim data files that must remain in `experiments/`.
- **Test**: Automated verification scripts belonging in `tests/`.
- **Generated report/data**: Markdown reports, JSON metrics, and manifests.
- **Existing user modification**: Tracked file modifications that must be preserved.
- **Unreviewed or unsafe**: Code with known defects, unhandled exceptions, or unverified claims.

### Summary Table

| Component / Path | Classification | Status & Assessment |
| :--- | :--- | :--- |
| `chakrview/runtime/session.py` | Existing user modification | **Sound**. Converts prompt assembly to standard ChatML format (`ConversationTemplate.format_conversation`). |
| `chakrview/training/sft_engine.py` | Core implementation | **Defect Identified**. Target-masking failure when all targets are `-100` returns `NaN` in `F.cross_entropy`. Lacks shape/boundary guard. |
| `experiments/batch39_dataset_builder/sft_batch_safety.py` | Stable implementation candidate | **Sound**. Comprehensive tensor validation for `SFTBatchSafetyGuard`. Currently isolated in experiment directory; needs engine-level integration. |
| `experiments/batch39_dataset_builder/corpus_pipeline.py` | Stable implementation candidate | **Sound**. Deterministic text chunker, SHA-256 verification, and train/val disjoint document splitter. |
| `experiments/batch39_dataset_builder/candidate_builder.py` | Experimental artifact | **Functional with Bottlenecks**. Successfully produced 46 deduplicated candidates; limited by template repetition in synthetic source files. |
| `experiments/batch39_dataset_builder/readiness_gate.py` | Stable implementation candidate | **Sound**. Enforces 8 strict validation gates before training authorization. |
| `tests/test_batch33_integrity.py` | Test | **Passes**. Verifies Git integrity and clean status rules. |
| `tests/test_batch34_sft_diagnostics.py` | Test | **Passes**. Validates EOS collapse diagnostics and token distribution. |
| `tests/test_batch35_recovery.py` | Test | **Passes**. Tests ChatML tokenization and generation stop criteria. |
| `tests/test_batch36_readiness.py` | Test | **Passes**. Verifies training-readiness gating. |
| `tests/test_batch37_independent_audit.py` | Test | **Passes**. Verifies document hash invariants and split independence. |
| `tests/test_batch38_audit_hardening.py` | Test | **Passes**. Verifies tamper detection and gate hardness. |
| `tests/test_batch39_dataset_builder.py` | Test | **Passes**. Verifies end-to-end dataset builder and safety guard. |

---

## 3. In-Depth Component Analysis

### 3.1 `chakrview/runtime/session.py`
- **Inspection**: The diff transitions manual string concatenation (`"User: ...\nAssistant:"`) to the structured `ConversationTemplate.format_conversation(messages, add_generation_prompt=True)`.
- **Integrity**: Imports `ConversationTemplate`, `ChatMessage`, and `Role` cleanly from `chakrview.runtime.conversation`. Matches the ChatML special tokens (`<|im_start|>`, `<|im_end|>`) introduced in Batch 35.
- **Finding**: High quality, non-breaking, properly integrated with conversation state.

### 3.2 SFT Engine & Target Masking Path (`chakrview/training/sft_engine.py`)
- **Inspection**: The forward pass computes:
  ```python
  logits = self.model(input_ids)
  loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
  ```
- **Vulnerability**: If all tokens in `targets` are set to `ignore_index` (`-100`), PyTorch `F.cross_entropy` produces `NaN`. In addition, out-of-bounds target IDs ($>= 4096$) trigger uncatchable CUDA/CPU memory indexing faults.
- **Remediation**: The engine boundary must execute a pre-forward safety check verifying:
  1. At least one target token is active ($\neq -100$).
  2. All token IDs fall within $[0, \text{vocab\_size}-1]$ (or are $-100$).
  3. No sequence exceeds `max_seq_len`.
  4. Input and target dimensions match exactly.

### 3.3 Batch 39 SFT Batch Safety Guard (`experiments/batch39_dataset_builder/sft_batch_safety.py`)
- **Inspection**: Implements `SFTBatchSafetyGuard.validate_batch(...)`. Checks:
  - Tensor shapes, dtype, and dimension matching.
  - Context length violations.
  - Active target token count ($> 0$).
  - Vocabulary range compliance ($0 \le id < 4096$ or $id == -100$).
  - EOS token supervision presence.
- **Finding**: The guard was written in Batch 39 but kept strictly inside `experiments/batch39_dataset_builder/`. It was never wired into `chakrview/training/sft_engine.py`. Integrating it directly into the training loop provides immediate defense-in-depth.

---

## 4. Test Suite Redundancy & Hard-Coded Invariants
- Across `tests/test_step*.py`, the canonical hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` is asserted in over 25 independent test files. This creates an unyielding invariant protecting the 3,443,136-parameter architecture.
- Tests from Batches 33–39 are independent, self-contained, and run in under 10 seconds total. They do not duplicate runtime models unnecessarily; rather, they employ lightweight fixtures and strict assertions.

---

## 5. Conclusion & Actionable Next Steps
1. **Commit Planning**: Formulate a 4-stage logical commit progression for Batches 33–39 that isolates runtime code, test suites, and audit tooling.
2. **Safety Guard Integration**: Wire `SFTBatchSafetyGuard` safely into `chakrview/training/sft_engine.py` with backward compatibility and zero overhead on standard batches.
3. **Data Pipeline Diagnosis**: Resolve why the raw passage extractor yielded only 46 deduplicated records out of 6,044 raw chunks.
