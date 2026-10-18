# INITIAL STATE CAPTURE: BATCH 2581–2620
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Capture Timestamp:** 2026-10-18T11:54:15+05:30  
**Local Machine:** Windows PowerShell / Python 3.14.7 (.venv_gpu)  

---

### 1. Git State & Starting Commit Discrepancy Note

- **Current Branch:** `dev/v0.6-v0.7-validation`
- **Expected HEAD Commit:** `114aa79fa0e0125d7815690020f913466ea978b5` (the Batch 2541–2580 deliverables commit)
- **Actual HEAD Commit:** `0570ff672ec04c2b48f3ca7b1f12901e95a8ca36`
- **Commit Discrepancy Analysis:**
  - The immediately preceding user request explicitly instructed: `"do commit all and push to github"`.
  - As part of executing that directive, all remaining uncommitted working tree files (historical sprint docs `docs/sprints/batch103-122`, experiment reports `experiments/batch83-122`, test suites `tests/test_batch103-118`, and `chakrview/runtime/session.py`) were committed into commit `0570ff6` right on top of commit `114aa79`, and then pushed to `origin/dev/v0.6-v0.7-validation`.
  - Therefore, `0570ff6` is the direct linear descendant of `114aa79`:
    - `0570ff6` parent is `114aa79`.
    - `114aa79` is completely intact and part of the active branch history.
  - This discrepancy is formally documented per Safety Requirement 7.

---

### 2. Working Tree & Protected File Status

- **Working Tree Cleanliness:** `git status --short` is completely clean (`nothing to commit, working tree clean`).
- **Protected File (`chakrview/runtime/session.py`):**
  - `git diff HEAD -- chakrview/runtime/session.py` is empty (0 diff lines).
  - The file exists at sha1 blob `79fbd61...` as committed in `0570ff6`.
  - Safety check confirmed: No modification, reset, staging, or overwriting of `chakrview/runtime/session.py` has occurred in this batch.

---

### 3. Primary Manifest, Model Checkpoint, and Tokenizer Hashes

| Asset Name | Relative Path | Size (bytes) | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Canonical Step 10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 42,212,787 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Seed 42 CPT Candidate Checkpoint** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | 41,402,267 | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` |
| **Split V2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | 45,759 | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` |
| **Cleaned Corpus V2 Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | 1,345,799 | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` |
| **Canonical Vocab BPE** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical Merges BPE** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | 76,299 | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |

---

### 4. Previous Batch Key Evidence Reports

| Report | Relative Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Batch 2541 Final Evidence** | `experiments/batch2541_2580/FINAL_EVIDENCE.json` | `95f63c921ecca30ec772725b55b3ca59eb71fa9d2b7d807cce1e44974b512ee8` |
| **Batch 2541 Training Coverage Final** | `experiments/batch2541_2580/TRAINING_COVERAGE_FINAL.json` | `b29793be138cc9ca43d03cbf07073bfddc517eaf0b030b4a1a0c98b8bc02952b` |
| **Batch 2541 Amended Preregistration** | `experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.json` | `0e695e13224112d641bfc552053eb47236cb8335c116f1e768bcaccb4e66efc9` |
| **Batch 2541 Held-Out Revalidation** | `experiments/batch2541_2580/HELDOUT_REVALIDATION.json` | `8954b5cf655776fd8cc7f8f9e2562785d1e312d055055d6099f85f7275e59058` |
