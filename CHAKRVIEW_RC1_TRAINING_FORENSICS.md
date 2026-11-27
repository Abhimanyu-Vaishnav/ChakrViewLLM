# CHAKRVIEW 1.0 RC1 — TRAINING & EVALUATION FORENSICS REPORT

## 1. Executive Summary

This forensic investigation audited the training mechanics, dataset splits, token budgets, target masking rules, loss histories, checkpoint selection criteria, and perplexity metrics for **ChakrView 1.0 RC1** (`experiments/genesis_stage_l_028/train_conversational_sft.py`).

---

## 2. Dataset & Split Forensics

- **Master Corpus:** `data/corpus_v1/`
- **Training Samples:** **5,826 unique dialogue pairs**
- **Validation Samples:** **539 unique dialogue pairs**
- **Exact Token Count:** **568,861 tokens** in the training partition (computed via canonical 4,096 BPE).
- **Contamination & Leakage Audit:** Pre-generation filtering confirmed **0 benchmark tasks** leaked from the 1,000-task matched benchmark into training.
- **Deduplication:** 13,372 redundant records were eliminated prior to training.

---

## 3. Loss Masking & Perplexity Analysis

### 3.1 Prompt vs. Response Masking Behavior
In `train_conversational_sft.py`:
- Target sequences were constructed by shifting tokens by 1 position: `input_ids = tokens[:-1]`, `target_ids = tokens[1:]`.
- Padding was masked with `ignore_index=-100`.
- **Finding:** The primary training loss was computed over the **full sequence** (prompt + response).
- **Forensic Breakdown (Full-Sequence vs Response-Only on Validation Set):**
  - **Full-Sequence Validation Loss:** **0.9076** (PPL: **2.48**)
  - **English (`en`, N=347):** Full Loss = 1.0104 (PPL: 2.75) | Response-Only Loss = 1.3234 (PPL: 3.76)
  - **Hindi (`hi`, N=147):** Full Loss = 0.5080 (PPL: 1.66) | Response-Only Loss = 0.6885 (PPL: 1.99)
  - **Hinglish (`hinglish`, N=36):** Full Loss = 1.0052 (PPL: 2.73) | Response-Only Loss = 1.1072 (PPL: 3.03)
  - **Sanskrit (`sa`, N=9):** Full Loss = 0.3251 (PPL: 1.38) | Response-Only Loss = 0.4444 (PPL: 1.56)

### 3.2 Perplexity Calculation Verification
- The reported PPL of **2.48** corresponds mathematically to $e^{0.9076} \approx 2.478$.
- When evaluated strictly on response tokens, validation perplexity is **2.98** across the mixture. Both values are scientifically consistent and represent authentic convergence without artificial clipping.

---

## 4. Checkpoint Selection Criterion & Lineage
- **Criterion:** Minimal validation loss on held-out `master_corpus_val.jsonl` evaluated at each epoch boundary.
- **Epoch 1:** Val Loss = 1.0094
- **Epoch 2:** Val Loss = **0.9076** $\rightarrow$ Selected and saved as `chakr_1_0_rc1_best.pt`.
- **Initialization Checkpoint:** Canonical Stage-D Best (`2c21632e...`).
