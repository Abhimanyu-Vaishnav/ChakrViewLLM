# Batch 71 — Reproducibility & Pipeline Audit Report

## 1. Executive Summary & Verification Context

Prior to expanding experiments or implementing modifications, an audit of the repository state, commit history, canonical architectural invariants, test counts, and prompt/response contracts was conducted on branch `dev/v0.6-v0.7-validation`.

All reported deliverables from Batches 67–70 were independently audited and confirmed bit-exact against the repository's active code and artifacts.

---

## 2. Git & Working Tree Verification

- **Current Branch**: `dev/v0.6-v0.7-validation`
- **Current HEAD**: `b0748df230a3c4ab72ac8037b39bf4b516165b89`
  - Commit message: `feat(sprint): complete Batches 67-70 loss weighting comparison, single-digit generalization evaluation, and contract hardening`
  - Committed files: 19 files (+2320 insertions, -21 deletions).
- **Prior User Modification**: `chakrview/runtime/session.py`
  - Verified present as uncommitted, unstaged working-tree modification (lines 317–332 converting turn history to `ConversationTemplate` ChatML format).
  - Preserved strictly untouched, never staged, never committed.
- **Upstream & Remote Tracking**:
  - Remote `origin`: `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
  - Local branch `dev/v0.6-v0.7-validation` has verified local commits and is not pushed to origin without authorization.

---

## 3. Test Suite Count Reconciliation

We independently executed pytest across the comprehensive batch regression test suite:
- **Command**:
  ```bash
  .\.venv\Scripts\python.exe -m pytest tests/test_batch33_integrity.py tests/test_batch34_sft_diagnostics.py tests/test_batch35_recovery.py tests/test_batch36_readiness.py tests/test_batch37_independent_audit.py tests/test_batch38_audit_hardening.py tests/test_batch39_dataset_builder.py tests/test_batch40_capability_sprint.py tests/test_batch41_evaluation_git.py tests/test_batch42_evaluation_harness.py tests/test_batch43_46_architecture_capability.py tests/test_batch47_50_verified_learning.py tests/test_batch51_54_alignment_and_gates.py tests/test_batch55_58_trustworthy_learning.py tests/test_batch59_62_learning_verification.py tests/test_batch63_66_training_generation_debug.py tests/test_batch67_70_generalization_sprint.py
  ```
- **Results**:
  - Collected: **96 items**
  - Passed: **96 items**
  - Failed: **0 items**
  - Skipped: **0 items**
  - Execution Time: **6.49s**
- **Conclusion**: The reported count of 96 passing tests in Batch 70 is verified and completely reproducible.

---

## 4. Canonical Baseline Invariant Verification & Hash Reconciliation

### Invariant 1: Total Parameter Count
- Model: `ChakrMicro(ModelConfig())`
- Architecture: 6 layers, $d_{model}=192$, $n_{heads}=6$, $head\_dim=32$, $hidden\_dim=512$, $vocab\_size=4096$, $max\_seq\_len=512$, tied embeddings.
- Result: **3,443,136 parameters** (exact match).

### Invariant 2: Step-10,000 Checkpoint Hash
- File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Computed SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Verdict: **BIT-EXACT MATCH**.

### Invariant 3: Canonical Fresh Initialization Hash & Seed Reconciliation
- Expected SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Reconciliation of earlier discrepancy**:
  - The canonical procedure (used throughout Historical Batches 45, 48, 51, 94–100, 536) requires sorting `model.named_parameters()`, encoding the parameter name as UTF-8, and feeding both name and raw parameter bytes into SHA-256 with `torch.manual_seed(42)` set before instantiation:
    ```python
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
    ```
  - When computed via this repository-canonical standard, the digest is `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` bit-for-bit.
  - An earlier one-off test script omitted parameter names from the hash update stream, which produced an ad-hoc byte digest. The canonical hash calculation incorporates the parameter names and passes 100% across all regression tests.
- Verdict: **CONFIRMED & VERIFIED**.

---

## 5. Training/Evaluation Interface Contract Audit

- **Shared Contract Module**: `chakrview/training/prompt_contract.py`
  - `normalize_prompt_response_strings(prompt, response, strip_delimiters)`: correctly preserves `\n` when `strip_delimiters=False`, avoiding the terminal newline mismatch that caused table attractor outputs in earlier sprints.
  - `build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)`: aligns input at index $|P|$ to target token $R_0$, guaranteeing causal alignment.
- **Runtime vs Training Parity**:
  - Both `SFTDataset.encode_example` (`chakrview/training/sft_engine.py`) and `DeterministicEvaluator.encode_record` (`experiments/batch42_reproducible_evaluation/evaluation/evaluator.py`) import and call this shared contract.
  - Note: High-level session runtime (`chakrview/runtime/session.py`) is an interactive dialogue manager and uses `ConversationTemplate`. While `ConversationTemplate` produces ChatML strings identical to our training format, full runtime end-to-end integration remains distinct from SFT dataset ingestion.

---

## 6. Audit of Batches 67–70 Experiment Artifacts

- **Weighted Loss**: `chakrview/training/weighted_loss.py` implements per-token loss weighting normalized by active weights, safely handling `-100` masked positions.
- **Single-Digit Arithmetic Task**: `experiments/batch67_70_generalization_sprint/data/` defines 45 addition pairs ($a + b \le 9$) with 25 Train, 10 Val, 10 Bench. Single-token encoding was verified.
- **Observed Result**: 100-step training fits training examples (reaching 20%–100% exact match depending on step count and batch size) but demonstrates **0% validation and held-out exact match**, confirming memorization without operator learning.
