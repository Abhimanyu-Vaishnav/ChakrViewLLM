# CANDIDATE INTEGRITY AUDIT

**Batch:** 2261–2300  
**Candidate Identifier:** `chakrmicro-v0.6-candidate-cpt-seed42-step125`  
**Candidate Manifest:** `experiments/batch2261_2300/candidate_manifest.json`  

---

## 1. Candidate Checkpoint Integrity

We audited the physical artifact representing the research candidate:
- **On-Disk Path:** `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt`
- **File Size:** 41,402,267 bytes
- **SHA-256 Checksum:** `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`
- **Model Loading Test:** Successfully loaded into memory using `ChakrMicro(canonical_config)` with `strict=True`. All 3,443,136 parameter tensors match the canonical architecture specification.
- **Independence from Baseline:** The candidate checkpoint is stored in an isolated experiment directory and does not overwrite or mutate `training_checkpoint_step_10000.pt`.

---

## 2. Release Boundary and Classification

| Dimension | Audit Finding | Policy Adherence |
| :--- | :--- | :---: |
| **Release Scope** | Internal research candidate (`v0.6.0-rc1`) | **PASS** |
| **Public Release Declared?** | **No.** Marked strictly `public_release: false` | **PASS** |
| **Baseline Checkpoint Status** | Step-10,000 checkpoint unchanged (`Delta W = 0`) | **PASS** |
| **Tokenizer Immutability** | Canonical `vocab_4096` unmodified | **PASS** |
| **Context Length Limit** | Frozen at canonical 512 tokens | **PASS** |
| **Reproduction Command** | Verified: `python experiments/batch2221_2260/run_independent_evaluation.py` | **PASS** |

---

## 3. Candidate Manifest Verification

A formal machine-readable specification `candidate_manifest.json` was generated in `experiments/batch2261_2300/`. It documents:
1. Exact checkpoint SHA-256 checksums for both baseline and candidate.
2. Architecture hash `c5571c9c...` and parameter count (3,443,136).
3. Verified test metrics (-0.1928 token CE delta, 76.55% document win rate).
4. Explicit list of known limitations (Classical Sanskrit drift, reliance on symbolic calculator for arithmetic).

**Candidate Integrity Verdict:** `PASS` (Research Candidate Validated; Production Baseline Protected).
