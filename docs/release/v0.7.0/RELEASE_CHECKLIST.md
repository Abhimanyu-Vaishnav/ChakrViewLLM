# CHAKRVIEW v0.7.0 RELEASE CHECKLIST & GATE AUDIT

## 1. Safety and Protected Artifacts Gate
- [x] `chakrview/runtime/session.py` is 100% untouched (`git diff --stat` is empty, 12,844 bytes).
- [x] Canonical Step 10,000 checkpoint is verified bit-exact: SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
- [x] Canonical checkpoint was NOT copied into the git worktree (gitignored `*.pt`).
- [x] Canonical Tokenizer verified bit-exact:
  - `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
- [x] ZERO training, fine-tuning, optimizer steps, or model weight mutations occurred ($\Delta W = 0$).

## 2. Test Suite & Regression Verification Gate
- [x] All 38 regression tests pass:
  - 10 AST-hardened security executor tests (`tests/test_security_hardened_executor.py`)
  - 5 coding benchmark runner tests (`tests/test_coding_suite.py`)
  - 4 intelligence harness audit tests (`tests/test_intelligence_harness_audit.py`)
  - 3 n-gram repetition blocking tests (`tests/test_sampling_ngram_blocking.py`)
  - 3 coding assistant tests (`tests/test_coding_assistant.py`)
  - 7 inference session tests (`tests/test_inference_session.py`)
  - 2 coding data pipeline leakage tests (`tests/test_coding_data_pipeline.py`)
  - 2 intelligence suite smoke/schema tests (`tests/test_intelligence_suite.py`)
  - 2 release smoke & tokenizer hash tests (`tests/test_v060_release_smoke.py`)
- [x] 3 skipped tests are exclusively weight-dependent tests in `test_v060_release_smoke.py` when running without the external checkpoint path flag.
- [x] 0 test failures.

## 3. Real Benchmark Verification Gate
- [x] 32-Task Native Intelligence Suite executed against real canonical weights:
  - Baseline greedy: 1/32 passed (3.1%), 17/32 attractor collapse loops (53.1%).
  - With `no_repeat_ngram_size=3`: 1/32 passed (3.1%), 0/32 attractor collapse loops (0.0%).
- [x] 12-Task Coding Benchmark executed against real canonical weights:
  - Functional pass rate: 2/12 (16.7%).
  - Syntax valid rate: 5/12 (41.7%).
- [x] 10 Conversational Probes executed against real canonical weights:
  - Empirically documented baseline text output in `evaluation/intelligence_suite/conversational_probes_canonical_step10000.json`.

## 4. Documentation & Release Packaging Gate
- [x] Project North Star and decision hierarchy documented in `docs/PROJECT_NORTH_STAR.md`.
- [x] Capability roadmap with version contracts up to v1.0 documented in `docs/CHAKRVIEW_CAPABILITY_ROADMAP.md`.
- [x] Release notes authored in `docs/release/v0.7.0/RELEASE_NOTES.md`.
- [x] Reproducibility guide authored in `docs/release/v0.7.0/REPRODUCIBILITY.md`.
- [x] Root `README.md` updated with v0.7.0 features and commands.
- [x] Root `CHANGELOG.md` updated with v0.7.0 entry.

## 5. Git & Release Boundary Gate
- [x] Feature branch `feat/v0.7-coding-foundation` based cleanly on v0.6.0 release commit `12e6e74bb3e08d0c595974a24d661ce19bae1e45`.
- [x] Clean staging: only reviewed release files, documentation, and evaluation artifacts staged.
- [x] No secrets, scratch files, or private environment variables included.