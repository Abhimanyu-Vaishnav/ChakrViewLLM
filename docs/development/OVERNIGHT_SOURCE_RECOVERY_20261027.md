# CHAKRVIEW OVERNIGHT CANDIDATE FORENSIC & SOURCE RECOVERY REPORT

**Audit Date:** 2026-10-27T22:45:00+05:30
**Lead Auditor:** Senior Git Repository Maintainer & Release Integrity Engineer
**Candidate Directory:** `experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/packaging_validation/test_fresh_user_space/ChakrView/`
**Repository Root:** `D:\Project\ChakrView`
**Branch:** `main` (commit: `2146f9d3cf36126efa3aa3408ab0391317a4fc8d`)

---

## 1. Candidate Provenance & Purpose

A thorough investigation of repository commit history, overnight run logs, and release documentation established the exact provenance and purpose of the candidate directory:

1. **Originating Task:** Overnight Release Candidate Readiness Mission (executed October 25, 2026, 21:33:37 UTC, documented in `V08_OVERNIGHT_RELEASE_EXECUTION_REPORT.md`).
2. **Operational Purpose:** The directory `packaging_validation/test_fresh_user_space/ChakrView/` is a **release-validation sandbox fixture**. It was constructed automatically to simulate an end-user downloading the package and verifying fresh wheel installation and anonymous public checkpoint download in a sterile workspace before publishing release `v0.8.0`.
3. **Source Snapshot Baseline:** The source tree inside `test_fresh_user_space/ChakrView/` was copied from development branch `dev/v0.6-v0.7-validation` at release candidate commit `ae96c645eb08c438440ffbe7546754341964a93e`.
4. **Epistemic Classification:** The candidate directory is a **Tier 2 validation fixture / snapshot**. It was **never intended to replace canonical `main`**, which represents the curated, production-tested release line governed by PR review gates.

---

## 2. Complete File-by-File Comparative Audit

A recursive cryptographic comparison was executed across all **1,721 files** in the candidate tree against the repository at `HEAD`:

| Comparative Category | File Count | Percentage | Detailed Findings & Description |
|---|:---:|:---:|---|
| **Identical to Tracked Repo (`main`)** | 849 files | 49.33% | Bit-exact byte matches with the active repository across `chakrview/`, `data/`, and runtime modules. |
| **Generated / Python Bytecode / Excluded** | 859 files | 49.91% | `__pycache__/*.pyc` files generated during the smoke test in the validation environment. |
| **New Files (Absent from `main`)** | 6 files | 0.35% | Specific modules present in `dev` branch or local validation fixture (see Section 3). |
| **Modified Variants (Differ from `main`)** | 7 files | 0.41% | Files reflecting developmental versions from branch `dev/v0.6-v0.7-validation` (see Section 4). |
| **Total Candidate Files Inspected** | **1,721 files** | **100.0%** | **Full inventory reconciled** |

---

## 3. Detailed Audit of the 6 "New" Files

| Relative Path | Size | SHA-256 Digest | Status & Analysis | Recommended Disposition |
|---|:---:|---|---|---|
| `smoke_test.py` | 962 B | `def7b4bdb0bf...` | Clean-environment sanity script testing `LocalModelRuntime.from_default().chat("Ram lives in Delhi.")`. | **RECOVER TO SCRIPT FIXTURES** (`scripts/smoke_test_fresh_user.py`) |
| `chakrview/brain/config_15m.py` | 1,780 B | `166433bf3645...` | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:chakrview/brain/config_15m.py` (commit `255537a`, Exp26 15M scaling hypothesis). | **Already tracked on `dev` branch**; quarantined from `main` pending 15M training. |
| `chakrview/evaluation/task_evaluator.py` | 36,815 B | `f812360324ee...` | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:chakrview/evaluation/task_evaluator.py`. Evaluates task extraction rubrics. | **Already tracked on `dev` branch**; archived on `archive/exp18-exp30-research-evidence`. |
| `chakrview/evaluation/__init__.py` | 264 B | `5937aecf38ca...` | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:chakrview/evaluation/__init__.py`. | **Already tracked on `dev` branch**. |
| `chakrview/export/export_chakrmicro_gguf.py` | 5,026 B | `4074cd367f45...` | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:chakrview/export/export_chakrmicro_gguf.py`. GGUF export utility. | **Already tracked on `dev` branch**; experimental export preview. |
| `chakrview/export/gguf_writer.py` | 5,072 B | `3d24d16a4ff4...` | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:chakrview/export/gguf_writer.py`. Indigenous binary GGUF format writer. | **Already tracked on `dev` branch**; experimental export preview. |

**Key Finding:** 5 of the 6 "new" files are **already tracked and versioned in Git** on the development baseline branch (`dev/v0.6-v0.7-validation`) and preserved in the research archive branch (`archive/exp18-exp30-research-evidence`). They were intentionally excluded from `main` under the **Branching & Merge Policy** because they represent experimental scaling hypotheses (15M config) and export previews (GGUF writer) that have not undergone formal release promotion gating.

---

## 4. Detailed Audit of the 7 "Modified" Files

| Relative Path | Candidate vs Repo | Detailed Forensic Comparison | Status & Analysis |
|---|:---:|---|---|
| `README.md` | 10.7 KB vs 14.2 KB | Candidate matches historical commit `12e6e74` (v0.6.0). Repo `main` contains updated v0.8.0 release preview documentation. | **Repo `main` is authoritative and newer.** Overwriting would regress documentation. |
| `requirements.txt` | 259 B vs 294 B | **100% Bit-exact match** with `dev/v0.6-v0.7-validation:requirements.txt`. Repo `main` has clean production dependency bounds. | **Repo `main` is authoritative.** |
| `chakrview/runtime/coding_assistant.py` | 6.4 KB vs 6.6 KB | **100% Bit-exact match** with `dev/v0.6-v0.7-validation`. Repo `main` contains subsequent hardening. | **Repo `main` is authoritative.** |
| `chakrview/runtime/neural_runtime_adapter.py` | 15.8 KB vs 17.4 KB | **100% Bit-exact match** with `dev/v0.6-v0.7-validation`. Repo `main` contains v0.8.0 integrity checks and hash allowlists. | **Repo `main` is authoritative.** |
| `chakrview/runtime/sampling.py` | 9.2 KB vs 11.2 KB | **100% Bit-exact match** with `dev/v0.6-v0.7-validation`. Repo `main` has enhanced temperature validation. | **Repo `main` is authoritative.** |
| `chakrview/training/prompt_contract.py` | 2.7 KB vs 2.0 KB | **100% Bit-exact match** with `dev/v0.6-v0.7-validation`. | **Repo `main` is authoritative.** |
| `chakrview/training/sft_engine.py` | 18.7 KB vs 16.5 KB | **100% Bit-exact match** with `dev/v0.6-v0.7-validation`. | **Repo `main` is authoritative.** |

**Conclusion:** In all 7 modified files, the candidate tree contains **historical development snapshots** from `dev/v0.6-v0.7-validation`. The current files on `main` contain necessary security hardening, hash verification allowlists, and release documentation. Overwriting `main` with the candidate files would cause regressions.

---

## 5. Eligible Source Recovery Action

The only genuinely unique source file absent from both `main` and `dev/v0.6-v0.7-validation` is:
- `smoke_test.py` (962 bytes): A clean user-space validation script testing `LocalModelRuntime.from_default().chat("Ram lives in Delhi.")`.

### Recovery Destination:
- Recovered cleanly to: `scripts/smoke_test_fresh_user.py`
- Added companion documentation: `docs/development/OVERNIGHT_SOURCE_RECOVERY_20261027.md`
