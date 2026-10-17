# ARTIFACT INTEGRITY RECHECK REPORT

**Batch:** 2421–2460  
**Audit Phase:** Phase 4 — Cryptographic Artifact Integrity Verification  
**Machine-Readable Evidence:** [`experiments/batch2421_2460/ARTIFACT_INTEGRITY_RECHECK.json`](file:///d:/Project/ChakrView/experiments/batch2421_2460/ARTIFACT_INTEGRITY_RECHECK.json)  
**Status:** **100% BIT-EXACT MATCH (VERIFIED)**  

---

## 1. Executive Summary & Audit Scope

Phase 4 mandates independent cryptographic verification (SHA-256) of all canonical checkpoints, manifests, and tokenizer assets against previously documented expected hashes, ensuring that neither evaluation nor coverage scripts referenced stale, altered, or corrupt artifacts.

---

## 2. Cryptographic Hash Comparison Matrix

| Artifact Description | Target Path | Expected SHA-256 | Recomputed Actual SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Step-10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **CPT Candidate Seed 42** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | **MATCH** |
| **Cleaned Corpus Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | **MATCH** |
| **Split V2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | **MATCH** |
| **Canonical Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Canonical Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |

---

## 3. Path Reference Integrity Verification

We inspected the evaluation scripts (`run_isolated_evaluation.py`) and coverage loaders to verify that they consume canonical paths rather than any temporary or stale copies:
- Evaluator references `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`: **VERIFIED**
- Evaluator references `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt`: **VERIFIED**
- Evaluator references `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`: **VERIFIED**
- Evaluator references `release/chakrview-v0.5.0/data/experiments/vocab_4096`: **VERIFIED**

---

## 4. Integrity Conclusion

All 6 core artifacts exist, are completely uncorrupted, and match their canonical digests bit-for-bit.
