# CR-EXP-18: Phase A — Baseline Recovery and Repository Integrity Report

## 1. Executive Status & Git Architecture
- **Repository Location:** `D:\Project\ChakrView`
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current Local HEAD:** `c7cee95f807f1fc67ff04b59a9edefa5b4512cf5`
- **Remote Tracking:** `origin/exp/cbr-01-conversational-breakthrough`
- **Remote HEAD:** `c7cee95f807f1fc67ff04b59a9edefa5b4512cf5`
- **Branch Synchronization:** Local and Remote are in exact 100% synchronization (`ahead 0, behind 0`).

## 2. Working Tree & Historical Change Investigation
- **Unstaged Changes Discovered:** 14 files in `experiments/cognition/cr_exp_17r/` show modified status.
- **Detailed Forensic Diff Check:** Direct `git diff` confirms that all 14 modifications consist strictly of single trailing newline formatting adjustments introduced by the Windows text editor upon file inspection.
- **Precautionary Safety Policy:** In strict adherence to Section 3 ("Do not discard existing changes to obtain a clean working tree"), these files were left completely intact and preserved. No `git checkout -- .` or reset operations were executed.

## 3. Cryptographic Invariant Verification Table

| Invariant Subject | Expected SHA-256 | Actual Measured SHA-256 | Verification Status |
| :--- | :--- | :--- | :---: |
| **Canonical Architecture Init** (`seed=42`) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (Bit-Exact)** |
| **Canonical Stage B Checkpoint** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (Bit-Exact)** |
| **Frozen Conversational Parent** (`cbr_sft_04`) | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Bit-Exact)** |
| **CR-17 Release Checkpoint** | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **PASS (Bit-Exact)** |
| **Protected Runtime** (`session.py`) | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Bit-Exact)** |

All non-negotiable safety rules and baseline invariants are 100% satisfied.
