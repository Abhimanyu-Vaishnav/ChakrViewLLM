# Batch 251: Initial Repository, Checkpoint, and Report Integrity Audit

**Milestone**: Batches 251–260  
**Date**: 2026-10-14  
**Workspace**: `D:\Project\ChakrView`  
**Development Branch**: `dev/v0.6-v0.7-validation`  
**Inspected Commit at HEAD**: `9cc3795` — `feat(batch241-250): integrate interactive intelligence and intent routing`  
**Parent Commit**: `a9574b4` — `feat(batch231-240): validate cross-attention and multi-hop grounding`

---

## 1. Baseline Invariants & Cryptographic Verification

| Artifact / Check | Expected Specification | Measured / Audited Value | Status |
|---|---|---|---|
| **Git Branch** | `dev/v0.6-v0.7-validation` | `dev/v0.6-v0.7-validation` | **VERIFIED** |
| **Commit at HEAD** | `9cc3795` | `9cc3795d50429f37df91cff205081726a330cf93` | **VERIFIED** |
| **Canonical Checkpoint SHA-256** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **INTACT** |
| **Canonical Parameter Count** | 3,443,136 | 3,443,136 parameters (ChakrMicro, 6 layers, $d_{model}=192$) | **INTACT** |
| **Protected File Integrity** | `chakrview/runtime/session.py` | Local unstaged working-tree changes preserved (lines 314–333) | **PRESERVED** |
| **Frozen Interactive Suite** | 21 items (`frozen_interactive_regression_suite.json`) | SHA-256: `2b225d598334a96a2358d801dab70a6723a30c6b4cce3c5ffcef675d7006697d` | **VERIFIED** |
| **Frozen 3-Hop Manifest** | 100 items (`frozen_threehop_manifest_100.json`) | SHA-256: `6bf47a2baf124859f63b9ed1da4e31777852146db3d12c6c34a112b58f96bfc3` | **VERIFIED** |

---

## 2. Reproduction of Previous Benchmark Results

### A. Frozen Interactive Regression Suite (Batch 242/249)
- Command: `python experiments/batch241_250_interactive_integration/run_interactive_regression.py`
- Result: **21/21 Passed (100.0%)**
- Execution time: ~5.2 seconds across all 21 test queries.
- Covers: Arithmetic, Pi constant, ALU definition, CPU explanation, 2-input addition code synthesis, Mars capital false premise, Hindi photosynthesis explanation, follow-up ("and What about ALU"), user name memory store/recall, and restart persistence.

### B. Three-Hop Grounding Evaluation (Batch 248)
- Command: `python experiments/batch241_250_interactive_integration/evaluate_threehop.py`
- Manifest: `frozen_threehop_manifest_100.json` (N=100)
- Checkpoint: `experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt`
- Reproduced Outcomes:
  - **Arm A (Cross-Attention Pointer + TokenTrie)**: **51/100 (51.0%) EM**, 17 distractor confusions, 20/20 correct abstentions on broken chains, average latency: 620.28 ms/item.
  - **Arm B (Deterministic Extractor)**: **20/100 (20.0%) EM**, 0 distractor confusions (succeeds solely on broken-chain abstention).
  - **Arm C (BM25 Lexical Grounding)**: **100/100 (100.0%) EM** (lexical overlap is trivially present in full context documents).

---

## 3. Hardware State & Detection Audit

- **Physical Discrete GPU**: NVIDIA RTX A1000 6GB Laptop GPU (PCIe Bus 01:00.0, Driver 597.16, CUDA 13.2 support).
- **Physical Integrated GPU**: Intel Iris Xe Graphics.
- **Active Python Environment**: Python 3.14.7 on Windows x64.
- **Active PyTorch Build**: `2.14.0+cpu` (`torch.cuda.is_available() == False`).
- **Audit Conclusion**: Because the installed PyTorch wheel in `.venv` is compiled for CPU only, **no PyTorch CUDA execution can occur without changing the binary**. In accordance with Rule 9, the environment will NOT be destructively modified. All computations and benchmarks must honestly report CPU execution.
