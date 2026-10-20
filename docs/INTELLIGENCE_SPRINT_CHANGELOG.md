# CHAKRVIEW INTELLIGENCE SPRINT v0.8 CHANGELOG

## 1. Scope & Constraints Adherence
- Target Worktree: `D:\Project\ChakrView-v07-coding-foundation`
- Source Repository: `D:\Project\ChakrView` (preserved read-only)
- Protected File: `chakrview/runtime/session.py` (UNTOUCHED, 0 diffs)
- Git Operations: ZERO commits, pushes, merges, tags, or rebases.
- Training: ZERO training runs, fine-tuning, optimizer steps, or weight updates executed.

---

## 2. Exact Files Changed / Created

### New Files Created:
1. `evaluation/intelligence_suite/intelligence_tasks.json`
   - 32 held-out evaluation tasks across 10 categories.
2. `evaluation/intelligence_suite/run_intelligence_benchmark.py`
   - Evaluation runner supporting deterministic evaluation, smoke verification, and JSON recording.
3. `evaluation/intelligence_suite/README.md`
   - Documentation and reproduction instructions.
4. `tests/test_intelligence_suite.py`
   - Unit tests verifying schema, coverage, and smoke-mode execution of the intelligence suite.
5. `tests/test_sampling_ngram_blocking.py`
   - Regression tests for n-gram repetition blocking and loop-breaking.
6. `docs/NEXT_TRAINING_EXPERIMENT_PROPOSAL.md`
   - Specification for 200-step copy-augmented SFT experiment (marked PROPOSED - NOT EXECUTED).
7. `docs/INTELLIGENCE_SPRINT_V08_REPORT.md`
   - Comprehensive sprint report with preflight, diagnosis, and results.
8. `docs/INTELLIGENCE_SPRINT_CHANGELOG.md`
   - This exact record of changes and verification.

### Modified Files:
1. `chakrview/runtime/sampling.py`
   - Added `apply_no_repeat_ngram(logits, tokens, ngram_size)`.
   - Added `no_repeat_ngram_size: int = 0` to `SamplingConfig`.
   - Wired n-gram blocking into `Sampler.sample()` for greedy and stochastic paths.
2. `chakrview/runtime/neural_runtime_adapter.py`
   - Integrated `no_repeat_ngram_size` into `NeuralRuntimeAdapter.__init__` and `generate()`.

---

## 3. Test & Verification Results

### Test Execution Summary:
- Command: `pytest tests/test_coding_data_pipeline.py tests/test_security_hardened_executor.py tests/test_coding_suite.py tests/test_coding_assistant.py tests/test_inference_session.py tests/test_v060_release_smoke.py tests/test_intelligence_suite.py tests/test_sampling_ngram_blocking.py`
- Result: **34 passed, 3 skipped, 0 failed** in 13.68s.
- Skipped tests:
  - `test_01_canonical_checkpoint_hash_integrity`: Checkpoint file absent from worktree (gitignored `*.pt`).
  - `test_04_model_loading_and_parameter_shapes`: Weight-dependent.
  - `test_05_deterministic_smoke_generation`: Weight-dependent.

### Verification Hashes:
- `vocab.json` SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Verified)
- `merges.json` SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Verified)
- Canonical checkpoint `training_checkpoint_step_10000.pt` in source: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified)