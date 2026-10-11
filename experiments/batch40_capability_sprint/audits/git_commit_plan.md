# Proposed Git Commit & Hygiene Plan: Batches 33–40

**Audit Date:** 2026-10-11  
**Target Branch:** `dev/v0.6-v0.7-validation`  
**Current HEAD:** `7c3be923acf997d98f693b3509bece86ddf97641`  
**Authorization Status:** **PROPOSAL ONLY — NO COMMITS AUTHORIZED OR EXECUTED**

---

## 1. Hygiene Policy & Principles

1. **Explicit Founder Review**: No Git write mutations (`git add`, `git commit`, `git push`, `git rebase`, `git merge`, `git reset`) will occur during this batch.
2. **Atomic Logical Units**: Changes accumulated across Batches 33–40 must not be dumped into a single monolithic commit. They are partitioned into 4 distinct, cohesive commits.
3. **No Large Artifacts in VCS**: PyTorch model weights (`.pt`), large dataset caches (`.arrow`, `.bin`), and verbose execution traces must remain uncommitted or tracked via `.gitignore`.
4. **Secret Scanning**: All files in the proposed commits have been inspected for hard-coded credentials, API keys, private tokens, and environmental secrets. None were found.

---

## 2. Proposed Commit Progression

### Commit 1: Runtime Session ChatML Formatting Integration
- **Commit Message**: `feat(runtime): integrate ConversationTemplate ChatML formatting into LocalModelSession`
- **Scope & Files**:
  ```
  chakrview/runtime/session.py
  ```
- **Rationale**: Formats multi-turn conversation prompts using `<|im_start|>` and `<|im_end|>` delimiters via `ConversationTemplate.format_conversation()`, establishing parity between inference runtime and ChatML-trained models.
- **Evidence**: Verified via `test_neural_inference.py` and multi-turn runtime tests. Zero syntax or typing errors.

---

### Commit 2: SFT Diagnostics, Recovery Tools, and Regression Tests (Batches 33–36)
- **Commit Message**: `feat(sft): add EOS collapse diagnostics, ChatML tokenization recovery, and test suites`
- **Scope & Files**:
  ```
  experiments/batch33_integrity/
  experiments/batch34_sft_root_cause/
  experiments/batch35_sft_recovery/
  experiments/batch36_sft_readiness/
  tests/test_batch33_integrity.py
  tests/test_batch34_sft_diagnostics.py
  tests/test_batch35_recovery.py
  tests/test_batch36_readiness.py
  ```
- **Rationale**: Captures the empirical root-cause analysis of SFT EOS collapse, token-level supervision auditing, and corresponding pytest verification suites.
- **Evidence**: All 4 test suites execute deterministically in $<5$s with zero failures.

---

### Commit 3: Dataset Builder, Corpus Pipeline, and Hardened Audit Gates (Batches 37–39)
- **Commit Message**: `feat(dataset): add reproducible corpus extraction pipeline, safety guards, and audit gates`
- **Scope & Files**:
  ```
  experiments/batch37_independent_audit/
  experiments/batch38_audit_hardening/
  experiments/batch39_dataset_builder/corpus_pipeline.py
  experiments/batch39_dataset_builder/candidate_builder.py
  experiments/batch39_dataset_builder/sft_batch_safety.py
  experiments/batch39_dataset_builder/readiness_gate.py
  tests/test_batch37_independent_audit.py
  tests/test_batch38_audit_hardening.py
  tests/test_batch39_dataset_builder.py
  ```
- **Rationale**: Implements deterministic SHA-256 corpus chunking, disjoint train/val splitting, near-duplicate sequence filtering, and SFT batch validation guards.
- **Evidence**: Verified against 27 canonical corpus documents and verified through 69 automated unit tests.

---

### Commit 4: SFT Engine Safety Integration & Batch 40 Capability Experiment
- **Commit Message**: `feat(training): integrate SFTBatchSafetyGuard into SFTEngine and add Batch 40 experiment suite`
- **Scope & Files**:
  ```
  chakrview/training/sft_engine.py
  experiments/batch40_capability_sprint/audits/
  experiments/batch40_capability_sprint/learning_experiment/
  experiments/batch40_capability_sprint/final_report.md
  experiments/batch40_capability_sprint/final_report.json
  tests/test_batch40_capability_sprint.py
  ```
- **Rationale**: Connects batch safety guards directly into the training loop, executes a controlled continuation training capability run, and establishes baseline weight invariance.
- **Evidence**: Full regression pass including `test_batch40_capability_sprint.py`.

---

## 3. Files to Remain Uncommitted / Excluded from Git

The following categories must NOT be added to Git:
1. **Model Checkpoints**: All `.pt` files under `experiments/batch*/` and `checkpoints/`.
2. **Intermediate Extraction Dumps**: Raw multi-megabyte JSONL files that can be regenerated deterministically via `corpus_pipeline.py`.
3. **Scratch Virtual Environments & Caches**: `__pycache__/`, `.pytest_cache/`, `.venv/`.
