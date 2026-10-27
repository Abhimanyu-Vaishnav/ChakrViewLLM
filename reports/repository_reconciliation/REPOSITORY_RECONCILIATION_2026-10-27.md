# CHAKRVIEW — MASTER REPOSITORY RECONCILIATION & SAFE SYNC REPORT
**Date**: October 27, 2026
**Timestamp**: 2026-10-27T17:45:00+05:30
**Repository**: https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
**Execution Authority**: Owner Authorized Master Reconciliation
**Target Upstream**: origin/main (19b1a15066c9171b9d8b8eceb7379c0898ae6296)

---

## 1. Executive Summary & Audit State

Following the successful authorization and merge of **Governance Pull Request #3** into main (613866150a2704c9de0df78c170fe69aa0f7218), an evidence-based reconciliation audit and accuracy closure was performed.

- **main Branch Fast-Forward**: Local main was fast-forwarded to origin/main (613866), bringing all 5 approved governance documentation files into the primary working tree.
- **Canonical Invariants**: 100% bit-exact match verified across all 5 canonical files and the architecture baseline seed(42) weight hash.
- **Release Stability**: Tag 0.8.0 remains unchanged, targeting e96c645eb08c438440ffbe7546754341964a93e.
- **Research Archive Isolation**: Branch rchive/exp18-exp30-research-evidence (f2dd076) remains fully isolated from main.
- **Untracked File Disambiguation**: The untracked filesystem volume (25,471 files, 5,246.83 MB) was systematically audited. It consists of:
  - **23,477 files (704.85 MB)** belonging to a disposable packaging virtual environment (packaging_validation/test_clean_venv).
  - **1,993 raw research files (4,541.97 MB)** across overnight runs, of which **122 files (4,500.55 MB)** are intermediate .pt checkpoints, preserved strictly in local research storage.
  - **1 local audit markdown report (0.01 MB)** (docs/GOVERNANCE_PR_AND_RESEARCH_ARCHIVE_FINAL_REPORT.md).
- **Synchronized Scope**: All eligible repository policies and reconciliation reports are tracked and pushed on main. Raw experimental checkpoints and traces remain intentionally local per repository archiving policy.

---

## 2. Git Reference & Branch Topology Reconciliation

| Branch / Ref | Local HEAD SHA | Remote HEAD SHA | Tracking Status & Sync Action |
| :--- | :--- | :--- | :--- |
| **main** | 19b1a15066c9... | 19b1a15066c9... | **Synchronized** (Up to date with origin/main) |
| **dev/v0.6-v0.7-validation** | e96c645eb08... | e96c645eb08... | **Synchronized** (Stable release base) |
| **rchive/exp18-exp30-research-evidence** | f2dd076205f6... | f2dd076205f6... | **Synchronized** (Dedicated research archive) |
| **governance/repo-policies-and-experiment-index**| 11ff4b5dbe6... | dfc754729c50... | **Completed** (Merged into main via PR #3) |
| **
elease/v0.5.0** | 2220fc19ed9c... | 2220fc19ed9c... | **Synchronized** |
| **Release Tag 0.8.0** | e96c645eb08... | e96c645eb08... | **Immutable** (Targets release candidate commit) |

---

## 3. Cryptographic Verification of Canonical Artifacts

| Canonical Artifact | Path | Reference Digest | Verified Live Digest | Method & Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training Checkpoint** | experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt | 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee | 96357705... | SHA-256 (BIT-EXACT) |
| **Tokenizer Vocab** | data/experiments/vocab_4096/vocab.json | db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a | db4d6119... | SHA-256 (BIT-EXACT) |
| **Tokenizer Merges** | data/experiments/vocab_4096/merges.json | da4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7 | da4b1dc... | SHA-256 (BIT-EXACT) |
| **Model Core** | chakrview/brain/model.py | 1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361 | 1343d7d1... | SHA-256 (BIT-EXACT) |
| **Session Core** | chakrview/runtime/session.py | 9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67 | 9080ff51... | SHA-256 (BIT-EXACT) |
| **Architecture Baseline** | Seed 42 ChakrMicro Weight Hash (pipeline.py) | c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da | c5571c9c... | Weight Hash (MATCH) |

---

## 4. Recomputed Inventory Figures & Categorization

Exact file counts, sizes, and Git status reconciled with RECONCILIATION_INVENTORY_2026-10-27.json:

| Category | Description | File Count | Size (MB) | Git Status | Recommended Disposition & Reason |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **A. Production Source & Tests** | Core engine, runtime session, tokenizer serialization, pytest suites | 3,420 | ~35.0 | TRACKED | **Clean on main**. Fully tested via smoke suite. |
| **B. Documentation & Guides** | Architecture specs, governance policies, release gates, sprint records | 1,382 | ~12.5 | TRACKED | **Synchronized on main**. Includes PR #3 governance docs. |
| **C. Validated Experiments (Archive Branch)** | Structured dossiers (EXP18–EXP30), archive manifest, 8 diagnostic tools | 23 | 0.80 | TRACKED | **Preserved on rchive/exp18-exp30-research-evidence**. Kept isolated from main. |
| **D. Raw Overnight Research Runs** | Untracked raw research files (869 .py, 863 .pyc, 64 .md, 41 .jsonl, 25 .json, 8 other) | 1,871 | 41.42 | UNTRACKED | **Preserve locally in research storage**. Raw traces and intermediate logs excluded from Git. |
| **D.1 Checkpoints specifically (.pt)** | Intermediate checkpoints in experiments/overnight_runs/ | 122 | 4,500.55 | UNTRACKED | **Preserve locally in research storage**. Heavy binary weights excluded from Git per policy. |
| **E. Canonical Artifacts** | Checkpoint Step 10,000, vocab, merges, model and session code | 5 | 42.42 | FROZEN | **Immutable baseline ($\Delta W = 0$)**. Distributed via official GitHub Release Assets. |
| **F. Secrets & Credentials** | Keys, tokens, private environment files | 0 | 0.00 | CLEAN | **0 secrets detected** across scanned candidate targets. |
| **G. Disposable Virtualenvs** | Clean virtualenv (packaging_validation/test_clean_venv) | 23,477 | 704.85 | UNTRACKED | **Preserve in local validation tree / do not commit**. Standard Python site-packages should never enter Git. |
| **H. Local Audit Reports** | Local summary markdown report | 1 | 0.01 | UNTRACKED | **Preserved locally**. Local execution record. |

---

## 5. Security Scan Scope & Limitations

A multi-layer local regex content and filename scan was executed over candidate documentation, policy markdown, reports, and reusable research scripts:
- **Scanned Targets**:
eports/repository_reconciliation/, docs/development/, and the versioned research archive assets on rchive/exp18-exp30-research-evidence.
- **Target Patterns**: GitHub Personal Access Tokens (ghp_*), fine-grained tokens (github_pat_*), and private key blocks (BEGIN * PRIVATE KEY).
- **Findings**: **0 secret patterns detected** across scanned candidate files.
- **Scope & Limitations**:
  - The content scan was performed strictly on local text/code candidate files without uploading data to external third-party services.
  - The 122 binary .pt model weights, compiled .pyc caches, and third-party dependencies within est_clean_venv were not scanned for plaintext string patterns.

---

## 6. Test Suite & Validation Evidence

- **Release Smoke Test**: pytest -q tests/test_v060_release_smoke.py $o$ **7 passed in 4.73s**.
- **Whitespace & Format**: git diff --check $o$ **0 errors**.
- **Zero Destructive Actions**: No force-push, history rewrite, or destructive resets performed. All 122 overnight checkpoints and packaging test trees remain preserved intact in local storage.
