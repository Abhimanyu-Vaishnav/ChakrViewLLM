# Baseline Reproduction & Audit Report
**Location:** `experiments/batch411_440_paraphrase/reports/BASELINE_REPRODUCTION.md`  
**Milestone:** Batches 411–440  
**Audit Target:** Commit `57fa583`, Canonical Checkpoint, and Candidate `candidate_step150_corrected_eos.pt`.

---

## 1. Verified Artifact Checksums & Status

| Artifact Description | Local File Path | Expected SHA-256 | Actual Measured SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED (Exact Match)** |
| **Candidate Checkpoint** | `experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt` | `27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e` | `27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e` | **VERIFIED (Exact Match)** |
| **Active Vocab / Tokenizer** | `release/chakrview-v0.5.0/data/experiments/vocab_4096` | N/A | Exists, 4,096 tokens, EOS=1, BOS=0 | **VERIFIED** |

---

## 2. Benchmark Reproduction Results

Re-executed `candidate_step150_corrected_eos.pt` on the diagnostic and fresh generalization sets:

1. **15-Question Diagnostic Benchmark:**
   - Previous Reported: **80.0%** (12 / 15 exact matches).
   - Re-evaluation: **12 / 15 (80.0%)** exact match, 100% clean EOS termination.
   - Status: **REPRODUCED IDENTICALLY**.

2. **Paraphrase Subset (3 items):**
   - Direct phrasing: 3/3 (100%).
   - Paraphrased phrasing: 1/3 (33.3%).
   - Status: **REPRODUCED IDENTICALLY**.

---

## 3. Analysis of Failure Modes & Bottleneck

- **Lexical Over-Reliance:** The model heavily conditions on surface n-grams (e.g. `"What is 14 plus 28?"` conditions strongly on `"14 plus 28"`, but `"Compute the sum of fourteen and twenty eight"` fails to trigger the addition path).
- **Narrow Training Phrasing:** The candidate was trained on only 1 template per concept (`"Question: ...\nAnswer:"`).
- **Research Goal for Batches 411–440:** Build a rigorous, multi-variant paraphrase augmentation curriculum that covers varied syntax, indirect clauses, and multi-step combinations across arithmetic, computer architecture, coding, science, and Hindi.
