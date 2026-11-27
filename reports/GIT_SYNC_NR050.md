# CHAKRVIEW GIT SYNCHRONIZATION AUDIT REPORT (NR-050)

**Date of Synchronization:** 2026-11-27  
**Repository Working Directory:** `D:\Project\ChakrView`  
**Remote URL:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`  
**Active Synchronized Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Summary of Changes Pushed

All legitimate project work from missions NR-020 through NR-049 was cataloged, staged, committed, and pushed to the remote repository.

| Metric | Recorded Value |
| :--- | :--- |
| **New Git Commit Hash** | `f75d23a` |
| **Commit Message** | `feat(nr-020-049): end-to-end neural training, forensics, RC2 remediation, and RC3 fast-track breakthrough` |
| **Files Committed & Pushed** | **104 files** (100,905 insertions) |
| **Remote Branch Tracking** | `origin/exp/cbr-01-conversational-breakthrough` (Up to date with `f75d23a`) |
| **Status of Working Tree** | Clean (`nothing to commit, working tree clean`) |

---

## 2. Inventory of Pushed Assets

1. **Source Code & Runtimes:**
   - `chakrview/app/brain_runtime.py` (Local neuro-symbolic brain runtime)
   - `chakrview/cli/interactive_chat.py` (Interactive conversational CLI)
   - `chakrview/brain/genesis/chakr_genesis_17m.py` (Dynamic RoPE sequence scaling fix)
2. **Datasets & Curations:**
   - `data/corpus_v1/` (Master corpus V1 scripts, manifests, and JSONL data)
   - `data/corpus_v2/` (Normalized prompt corpus V2 scripts, manifests, and JSONL data)
   - `data/corpus_v3/` (Semantic anchor mixture corpus V3 manifests and JSONL data)
   - `data/stage_j_020/` & `data/stage_k_021/` (1,000-task benchmark and ablation datasets)
3. **Experiment Suites:**
   - `experiments/nr045_generation_forensics/` (Tokenizer roundtrip, loss masking, and dataset audit scripts)
   - `experiments/genesis_stage_rc2_046/` (Response-masking unit tests, training pipeline, decoding logs)
   - `experiments/fast_track_047_049/` (Sealed 120 benchmark, RC1/RC2/RC3 evaluation scripts and JSON scores)
4. **Research Reports (30+ Documentation Deliverables):**
   - NR-020 to NR-028 series reports.
   - NR-045 forensics series (`NR045_NEURAL_GENERATION_FORENSICS.md`, `NR045_TOKENIZER_FORENSICS.md`, etc.).
   - NR-046 remediation series (`NR046_DATASET_V2_REPORT.md`, `NR046_TRAINING_REPORT.md`, `NR046_FINAL_REPORT.md`, etc.).
   - NR-047 to NR-049 fast-track reports (`NR047_049_FAST_TRACK_REPORT.md`, `NR049_RC2_COMPARISON.md`).
5. **Project Management:**
   - Updated `CHAKRVIEW_CURRENT_STATE.md`, `CHAKRVIEW_MISSION_LEDGER.md`, `CHAKRVIEW_ARTIFACT_REGISTRY.json`.

---

## 3. Deliberately Excluded Files (and Precise Reasons)

1. **Model Weight Checkpoints (`*.pt`):**
   - Files like `chakr_1_0_rc1_best.pt`, `chakr_1_0_rc2_best.pt`, and `chakr_1_0_rc3_best.pt` (~68MB each) are matched by `.gitignore` (`*.pt`) to prevent repository bloat and respect GitHub blob size recommendations without unverified LFS costs. Checkpoints remain locally preserved with cryptographically recorded SHA-256 hashes.
2. **Virtual Environments (`.venv/`, `.venv_gpu/`):**
   - System/hardware-specific binaries excluded per standard Python best practices.
3. **Temporary Caches:**
   - Python bytecode (`__pycache__/`) and local editor scratch files excluded.

---

## 4. Final Verification

`git status` confirms:
`## exp/cbr-01-conversational-breakthrough...origin/exp/cbr-01-conversational-breakthrough`
Zero unpushed commits. Working tree is completely clean and synchronized.
