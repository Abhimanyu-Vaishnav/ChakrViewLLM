# CHAKRVIEW — GOVERNANCE PR REVIEW & RESEARCH ARCHIVE COMPLETION REPORT
**Date**: October 27, 2026
**Repository**: `D:\Project\ChakrView` (`https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`)
**Status**: `GOVERNANCE PR CREATED (AWAITING OWNER MERGE APPROVAL)` & `RESEARCH ARCHIVE BRANCH PUSHED`

---

## 1. Executive Summary

All phases of the Execution Master Prompt have been systematically executed in strict compliance with the **Global Safety Rules**:
1. **Canonical Verification**: 100% bit-exact SHA-256 match across model checkpoint, vocabulary, tokenizer merges, model core, and session runtime.
2. **Governance Documentation**: Audited, cleaned of trailing whitespace, calibrated with epistemic boundary definitions, verified against pytest smoke suite (7/7 passed), and pushed.
3. **Governance Pull Request Created**: **PR #3** has been formally opened against `main` on GitHub:
   - **URL**: [https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3](https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3)
   - **Source Branch**: `governance/repo-policies-and-experiment-index` (`9f0c205f368167e3c746ce15f143b22ce121831e`)
   - **Base Branch**: `main` (`d0ec3c24ed46c3987733f3464be5a052cf42dbca`)
   - **File Diffs**: Exactly 5 governance documentation files changed (0 model weights, 0 code files).
   - **MERGE AUTHORIZATION BOUNDARY**: In adherence to the mandatory stop condition, **PR #3 has NOT been merged automatically**. It requires explicit owner approval.
4. **Research Archive Branch Created & Pushed**:
   - **Branch**: `archive/exp18-exp30-research-evidence`
   - **Remote SHA**: `3f13ccd1f6250e2605f1bb81c363f04db8ac50e0`
   - **Base SHA**: `ae96c645eb08c438440ffbe7546754341964a93e` (`dev/v0.6-v0.7-validation`)
   - **Scope**: Dedicated research evidence dossiers (`EXP18.md` through `EXP30.md`), archive manifest, index, and 8 reusable diagnostic scripts.
   - **Zero merges to `main`**: This research archive branch is permanently isolated from `main`.

---

## 2. Remote References & Commit Topology

| Reference | Local Commit SHA | Remote Commit SHA | Status |
| :--- | :--- | :--- | :--- |
| `main` | `d0ec3c24ed46c3987733f3464be5a052cf42dbca` | `d0ec3c24ed46c3987733f3464be5a052cf42dbca` | Synchronized / Frozen |
| `dev/v0.6-v0.7-validation` | `ae96c645eb08c438440ffbe7546754341964a93e` | `ae96c645eb08c438440ffbe7546754341964a93e` | Synchronized |
| `governance/repo-policies-and-experiment-index` | `9f0c205f368167e3c746ce15f143b22ce121831e` | `9f0c205f368167e3c746ce15f143b22ce121831e` | Synchronized (Ahead 1 vs `main`) |
| `archive/exp18-exp30-research-evidence` | `3f13ccd1f6250e2605f1bb81c363f04db8ac50e0` | `3f13ccd1f6250e2605f1bb81c363f04db8ac50e0` | Synchronized (Ahead 1 vs `dev`) |
| Release Tag `v0.8.0` | `ae96c645eb08c438440ffbe7546754341964a93e` | `ae96c645eb08c438440ffbe7546754341964a93e` | Immutable |

---

## 3. Cryptographic Verification of Canonical Artifacts

| Canonical Asset | Path | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| Canonical Checkpoint (3.44M) | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| Vocabulary | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| BPE Merges | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |
| Model Core Architecture | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **MATCH** |
| Runtime Protected Session | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |

---

## 4. Governance Pull Request Details (PR #3)

- **PR URL**: [https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3](https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3)
- **Title**: `docs(governance): establish branch, release, and research archive policies`
- **Base / Head**: `main` (`d0ec3c2`) $\leftarrow$ `governance/repo-policies-and-experiment-index` (`9f0c205`)
- **Files Modified (5 files)**:
  1. `docs/development/BRANCHING_AND_MERGE_POLICY.md`
  2. `docs/development/CANONICAL_ARTIFACTS.md`
  3. `docs/development/EXPERIMENT_ARCHIVING_POLICY.md`
  4. `docs/development/RELEASE_GATES.md`
  5. `docs/research/EXPERIMENT_INDEX.md`
- **Test Validation**:
  - `.\.venv\Scripts\python.exe -m pytest tests/test_v060_release_smoke.py -v`: 7 passed in 5.19s.
  - `git diff --check`: 0 errors.

---

## 5. Research Evidence Archive Coverage (Exp18–Exp30)

Branch: `archive/exp18-exp30-research-evidence` (`3f13ccd1f6250e2605f1bb81c363f04db8ac50e0`)

| Experiment ID | Empirical Focus & Epistemic Boundary | Versioned Dossier | Archive Artifact Status |
| :--- | :--- | :--- | :--- |
| **Exp18** | Aligned SFT & short scientific reasoning. 20/20 rejections formalize reasoning limits of 3.44M core. | `docs/research/experiments/EXP18.md` | `COMMITTED` (Dossier, Reconciled CSV) / `LOCAL_ONLY` (Traces) |
| **Exp19** | Debiased SFT & template concentration. Template loops mitigated; length collapse identified. | `docs/research/experiments/EXP19.md` | `LOCAL_ONLY` (Audit & Run Reports) |
| **Exp20** | Single-task contextual binding. High in-distribution copy (>85%); degraded on unseen distractors. | `docs/research/experiments/EXP20.md` | `LOCAL_ONLY` (Forensic Audits) |
| **Exp21** | Synthetic recombined copying. Bounded short-span improvements (<4 tokens); unstable on long spans. | `docs/research/experiments/EXP21.md` | `COMMITTED_IN_GIT` / `LOCAL_ONLY` (Weights) |
| **Exp22** | Attention copy loss treatment vs control. Faster attention convergence; no final gain over CE. | `docs/research/experiments/EXP22.md` | `COMMITTED_IN_GIT` / `LOCAL_ONLY` (Metrics) |
| **Exp23** | Single-token copying diagnostic. Isolated single-token copy failure to softmax vocabulary competition. | `docs/research/experiments/EXP23.md` | `COMMITTED_IN_GIT` / `LOCAL_ONLY` (Reports) |
| **Exp24** | Copying vs routing diagnostic. Routing accurate (>90%); copying failed on out-of-vocab tokens. | `docs/research/experiments/EXP24.md` | `COMMITTED_OR_READY_FOR_ARCHIVE` |
| **Exp25** | Vocabulary generalization diagnostic. Quantified sharp drop from 88.4% (seen) to 12.1% (unseen). | `docs/research/experiments/EXP25.md` | `COMMITTED_OR_READY_FOR_ARCHIVE` |
| **Exp26** | 15M architecture configuration & latency readiness. Hardware/CPU benchmark; not trained capability. | `docs/research/experiments/EXP26.md` | `COMMITTED_IN_GIT` / `LOCAL_ONLY` (Logs) |
| **Exp27** | Forensic pointer audit & gradient diagnostics. Found prefix leakage; vanishing pointer gradients. | `docs/research/experiments/EXP27.md` | `COMMITTED` (Diagnostic Tools) / `LOCAL_ONLY` (Audits) |
| **Exp28** | Fair controlled baseline vs pointer comparison. No statistically significant gain (64.2% vs 65.1%). | `docs/research/experiments/EXP28.md` | `COMMITTED` (Comparison Tool) / `LOCAL_ONLY` (Traces) |
| **Exp29** | Independent replication audit. 100% blind reproduction of Exp28 under randomized distractor noise. | `docs/research/experiments/EXP29.md` | `LOCAL_ONLY` (Audit Dossier & Metrics) |
| **Exp30** | Human review closure, release gate matrix & CPU smoke test. Formal closure of v0.8.0 release gates. | `docs/research/experiments/EXP30.md` | `COMMITTED` (Review Tools) / `LOCAL_ONLY` (Gate Logs) |

### Versioned Diagnostic & Research Tools (`scripts/research_tools/`)
1. `audit_leakage_and_splits.py`: Dataset split and leakage detection.
2. `test_gradient_diagnostic.py`: Gradient flow diagnostic harness for dual-head architectures.
3. `run_exp28_fair_comparison.py`: Controlled benchmark runner with randomized distractor positioning.
4. `run_reproduction.py`: Blind independent evaluation replication script.
5. `recompute_metrics.py`: Metric recomputation engine under varying distractor distance constraints.
6. `run_bounded_pilot.py`: Lightweight pilot training and evaluation harness.
7. `build_review_dossier.py`: Automated review queue generator for blind human review.
8. `reconcile_human_reviews.py`: Consensus reconciliation engine for multi-reviewer evaluation queues.

---

## 6. Actionable Owner Decisions & Next Steps

1. **Owner Governance PR Approval**:
   - Review GitHub PR **#3**: [https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3](https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/pull/3)
   - When satisfied, execute standard merge into `main`.
2. **Preserve Research Archive Branch**:
   - The branch `origin/archive/exp18-exp30-research-evidence` will remain indefinitely as the primary research evidence branch.
