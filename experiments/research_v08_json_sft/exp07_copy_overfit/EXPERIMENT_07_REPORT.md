# ChakrView Experiment 07 — Small-Set Copying Overfit Test Report

**Date**: 2026-10-21  
**Commit/Parent Checkpoint**: Canonical Parent Checkpoint `Step-10000` (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`)  
**SHA-256 (Parent Checkpoint)**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**SHA-256 (Canonical Tokenizer Vocab)**: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`  
**Execution Environment**: Isolated Python 3.14 virtual environment (`.venv`), CPU execution.  
**Experiment Directory**: `experiments/research_v08_json_sft/exp07_copy_overfit/`  

---

## 1. Executive Summary & Primary Status

### Primary Status
`TRAINING MEMORIZATION WITHOUT UNSEEN GENERALIZATION`

### Core Result
When exposed to repeated optimization on a fixed, tiny dataset of 16 single-token examples using an unweighted cross-entropy loss and standard AdamW optimizer, **ChakrMicro rapidly and perfectly fits the training examples, achieving 100.0% teacher-forced first-token accuracy and 100.0% free-generation exact match by Step 50 (loss dropping from 11.32 to 0.029 at Step 100)**.

However, when tested on 25 untouched unseen single-token values in the identical prompt template, the model achieves **0.0% (0/25) teacher-forced first-token accuracy** and **0.0% (0/25) free-generation exact match**. The mean rank of the target token on unseen values is **1,215.0 / 4,093** (probability $0.002\%$). Instead of executing an algorithmic copy circuit from context, the model routes unseen inputs to memorized training tokens (principally `'balan'` and `'bran'`).

---

## 2. Integrity and Pre-Training Audit

All pre-training safety gates and unit tests were verified before initiating optimizer steps:
1. **Parent Checkpoint Integrity**: Confirmed exact SHA-256 hash match: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
2. **Tokenizer Integrity**: Confirmed exact SHA-256 hash match on `vocab.json` (`db4d6119...`) and `merges.txt` (`ada4b1dc...`).
3. **Causal Alignment Verification**: Confirmed in [test_integrity_and_alignment.py](file:///d:/Project/ChakrView/experiments/research_v08_json_sft/exp07_copy_overfit/test_integrity_and_alignment.py) that prompt tokens are masked with `-100`, that logit at position `len(prompt_tokens)` correctly supervises the first response token, and that trailing EOS is predicted as a separate subsequent step.
4. **Disjointness Assertion**: Proved cryptographically and programmatically that `tiny_train_16.jsonl` (16 values) and `unseen_test_25.jsonl` (25 values) have zero set overlap.

---

## 3. Dataset Specification

| Dataset Split | Records | Response Token Count | SHA-256 Hash |
| :--- | :---: | :---: | :--- |
| `tiny_train_16.jsonl` | 16 | 1 token per response | `eb6b013d1dec19b76a47d458b9c2e851af4516ffb4408c3952e55b1470ccbdd2` |
| `unseen_test_25.jsonl` | 25 | 1 token per response | `07d30beb92279941b6ebc86c19e9d0ac6ebbb27ebd3e6bccda7e713ef09c4859` |

**Prompt Template**:
```text
Target value: {value}
Copy: 
```
**Response Format**: `{value}` (where `{value}` encodes to a single BPE token ID).

---

## 4. Training Dynamics

- **Optimizer**: AdamW ($lr = 5 \times 10^{-4}$, $\beta_1=0.9, \beta_2=0.999$, weight decay = 0.01, grad clip norm = 1.0)
- **Batch Size**: 4 examples per step (1 full epoch over 16 examples every 4 steps)
- **Loss**: Unweighted CrossEntropyLoss (`ignore_index=-100`)
- **Evaluation Interval**: Every 25 optimizer steps on the complete 16-example training set.

| Step | Train Loss | Teacher-Forced 1st-Token Acc (Train) | Free-Gen Exact Match (Train) | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **0** (Parent) | *N/A* | 0.0% (0/16) | 0.0% (0/16) | Target rank ~1477.9, prob 0.005% |
| **25** | 11.3244 | 56.2% (9/16) | 56.2% (9/16) | Rapid acquisition of train prompt/response binding |
| **50** | 1.5920 | **100.0% (16/16)** | **100.0% (16/16)** | Perfect copying of all 16 training values |
| **75** | 0.1972 | **100.0% (16/16)** | **100.0% (16/16)** | Loss collapses towards zero |
| **100** | **0.0294** | **100.0% (16/16)** | **100.0% (16/16)** | Early stopping triggered (loss < 0.05, 100% EM) |

---

## 5. Quantitative Evaluation Results

Evaluation was performed using the best checkpoint (`exp07_best_checkpoint.pt`, Step 100) and evaluated separately on the training set and the unseen held-out set. Value-copying accuracy is strictly separated from EOS accuracy.

| Split | Checkpoint | Teacher-Forced 1st-Token Acc | Free-Gen Exact Match | Mean Target Rank | Mean Target Prob | Output Diversity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tiny Train (16)** | Step-10000 (Parent) | 0.0% (0/16) | 0.0% (0/16) | 1,477.9 / 4,093 | 0.005% | 0.50 |
| **Tiny Train (16)** | **Exp07 Best (Step 100)** | **100.0% (16/16)** | **100.0% (16/16)** | **1.0 / 4,093** | **99.32%** | **1.00** |
| **Unseen Test (25)** | Step-10000 (Parent) | 0.0% (0/25) | 0.0% (0/25) | 1,220.4 / 4,093 | 0.004% | 0.48 |
| **Unseen Test (25)** | **Exp07 Best (Step 100)** | **0.0% (0/25)** | **0.0% (0/25)** | **1,215.0 / 4,093** | **0.002%** | **0.24** |

### Output Diversity & Failure Mode Analysis on Unseen Inputs

When presented with unseen values in the prompt, the model does not generate generic pretraining text (`"The Special of the Ar"`). Instead, the model's output collapses onto the newly memorized training tokens:
- 18 / 25 unseen inputs generated `"balan"` (Token ID 1278, which was one of the 16 training values).
- 4 / 25 unseen inputs generated `"bran"` (Token ID 2315, another training value).
- 1 / 25 generated `"rect"` (Token ID 1178, another training value).
- 1 / 25 generated `"tive"` (Token ID 1421, a training value fragment).
- 1 / 25 generated `"ative"` (Token ID 2689, a training value).

Total unique outputs across 25 unseen prompts: **6** (diversity ratio 0.24).

---

## 6. Interpretation

### What this Result Establishes
1. **ChakrMicro is fully capable of fitting the exact mapping**: The model parameter capacity (3.44M parameters) is more than sufficient to overfit and output the target values with near-100% confidence ($P = 99.32\%$, rank 1.0, 16/16 exact match) when given sufficient gradient steps (50–100 steps).
2. **Exposure bias is not the primary roadblock**: In the training set, teacher-forced first-token accuracy and free-generation exact match are identical ($100.0\% = 100.0\%$). When the first token logit is correct, autoregressive generation succeeds completely.
3. **The learned circuit is associative memory, not an induction copy circuit**: SFT on the tiny set learned an associative lookup between the specific prompt token sequences and the specific target tokens. It did **not** learn the generalized contextual copying operation ("attend to position $k$, route token at $k$ to position $t$").

### What this Result Does NOT Establish
1. **Does not prove general copying is impossible in transformers**: Standard transformers can form induction heads.
2. **Does not prove ChakrMicro can never learn copying under any causal SFT regime**: It shows that on a tiny dataset (16 examples), gradient descent minimizes loss by memorizing input-output associations rather than finding the generalized induction circuit. Learning generalized induction typically requires either architectural inductive biases (such as pointer mechanisms or explicit copy heads) or broad synthetic copying curricula across thousands of varied permutations.

---

## 7. Artifact Integrity

- **Parent Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified unchanged)
- **Canonical Vocab SHA-256**: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Verified unchanged)
- **Canonical Merges SHA-256**: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Verified unchanged)
- **Exp07 Best Checkpoint SHA-256**: `13a0d6da3fa0030753f14e6932ec0fa496da95b569bc117ead4f0742678dc33d`
- **Exp07 Final Checkpoint SHA-256**: `bf83a1c01f8367d8823ddcfaeaf7c1199f774779780c6d475feab3ff27a911d9`

---

## 8. Recommended Next Step

**Recommendation**: Conduct a controlled **Synthetic Copying Curriculum Experiment (Experiment 08)** across a large token-permutation set (e.g., 2,000–5,000 diverse random token copy pairs) using a standardized induction head training task, to determine whether causal SFT can induce generalized copy circuits when memorization of fixed values is made suboptimal by sample variety. If generalized copying still fails across a large synthetic dataset, that will provide empirical justification for architectural interventions (such as a pointer head).
