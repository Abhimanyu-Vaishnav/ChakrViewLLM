# CHAKRVIEW v0.8: EXPERIMENT 06.1 EVALUATION INTEGRITY AUDIT REPORT
## Audit of Reported Teacher-Forced Accuracy vs. First-Token Target Rank Discrepancy

- **Author**: ChakrView Research & ML Engineering Agent
- **Execution Date**: 2026-10-21
- **Experiment Identifier**: `research_v08_json_sft_exp06_1_evaluation_audit`
- **Output Directory**: `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/`
- **Audit Target**: Reconcile Experiment 06's reported Stage A training teacher-forced token accuracy of **48.0%** with a reported mean first-token target rank of **1,009.9 / 4,096** and mean target probability of **0.037%**.
- **Per-Example Audit Artifacts**:
  - `AUDIT_06_1_RESULTS.json` (SHA-256: `34784e2bc81b07b0266dd8dafaa84f8a96bd2b9a30b0573c418132050df6e59b`)
  - `AUDIT_06_1_PER_EXAMPLE.csv` (SHA-256: `b152c1a9e6c3bb849e9472b60e991d28f5e547724fa890c004b33adab8f715ba`)

---

## 1. Executive Summary & Root Cause Reconciliation

The audit evaluated all 180 records across Train ($N=100$), Validation ($N=30$), and Test ($N=50$) from the exact saved checkpoint `exp06_best_checkpoint.pt`.

### The Core Discovery:
1. **The Discrepancy Was Caused by Conflating First-Token Target Accuracy with Combined Response Sequence Accuracy (which included EOS)**:
   - In Stage A, the target value is a single token $T_1$, followed immediately by the mandatory `<EOS>` boundary token (Token ID 1).
   - In `evaluate_exp06.py` (lines 131–136):
     ```python
     for i in range(resp_start_idx, len(tf_input)):
         tgt = tf_target[i]
         pr = int(torch.argmax(logits[i, :]).item())
         resp_matches.append(pr == tgt)
     ```
     This loop iterated across **all** unmasked response positions, which for Stage A consisted of **2 tokens per example**:
     - Position 0 (`resp_start_idx`): Target is the value token $T_1$.
     - Position 1 (`resp_start_idx + 1`): Target is `<EOS>` (Token 1).
2. **Decomposition of the 48.0% Reported Number**:
   - Out of 50 Stage A training examples, the number of target-value tokens correctly predicted at Position 0 ($T_1$) was **0 / 50 (0.0%)**.
   - The number of `<EOS>` boundary tokens correctly predicted at Position 1 (given $T_1$ via teacher-forcing) was **48 / 50 (96.0%)**.
   - Combining both tokens across all 50 examples:
     $$\text{Combined Token Accuracy} = \frac{0 + 48}{50 + 50} = \frac{48}{100} = \mathbf{48.0\%}$$
3. **Reconciliation with Target-Token Rank and Probability**:
   - The reported mean target-token rank (**1,009.86 / 4,096**) and mean probability (**0.0374%**) were measured **strictly at Position 0** (the first token of the value).
   - At Position 0, the count of examples where target rank equaled 1 was **0 / 50 (0.0%)**.
   - Therefore, the metrics do **not** contradict each other. The model had **0% accuracy on copying the target value**, but **96% accuracy on emitting EOS after being fed the target value**.
   - Labeling the combined number as "Stage A Teacher-Forced Token Accuracy" without explicitly separating the content token from the EOS delimiter obscured the fact that value copying was completely unlearned.

---

## 2. Cryptographic Lineage & Checkpoint Verification

All artifacts were verified bit-exact against historical baselines:

| Artifact Role | File Path | Verified SHA-256 Checksum | Invariant Status |
| :--- | :--- | :--- | :--- |
| **Canonical Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Unmodified |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Unmodified |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Unmodified |
| **Exp06 Evaluated Checkpoint** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/checkpoints/exp06_best_checkpoint.pt` | `d81953f82b2a485102a47662e9eac038e4573b535dbe68dde0d9adebd60822d0` | Verified single checkpoint |
| **Exp06 Train Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/train.jsonl` | `aca94c894a07ef78f68401ee55705d894453ab3bf6c795b38c234ef465679ce5` | Unmodified |
| **Exp06 Val Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/val.jsonl` | `c09070c880475668c3495458311aab22de5417b119addb14e3dac4e79fa2939b` | Unmodified |
| **Exp06 Test Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/test.jsonl` | `1b536cb2ad1d462db814f164287e2974ff6b1948d006f89bc33cc55145ab276f` | Unmodified |

All metrics in Experiment 06 and Experiment 06.1 were computed from the **exact same checkpoint** (`exp06_best_checkpoint.pt`, step 110).

---

## 3. Full Recomputed Metrics Across All 180 Records

Recomputed directly from per-example model outputs using numerically stable softmax and exact 1-based descending rank sorting (breaking ties conservatively by min-index):

### Recomputed Breakdown Table:

| Split | Stage | Total Examples ($N$) | Free-Gen Exact Match | First-Token TF Accuracy (Target Token 0) | Target Token 0 Rank = 1 Count | Mean First-Token Rank | Mean First-Token Prob | Subwords TF Accuracy (Excluding EOS) | EOS TF Accuracy (Token 1) | Combined TF Accuracy (Reported in Exp 06) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **Stage A** (Single) | 50 | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **0** | **1,009.86** | **0.037%** | **0 / 50 (0.0%)** | **48 / 50 (96.0%)** | **48 / 100 (48.0%)** |
| **Train** | **Stage B** (Multi) | 50 | **0 / 50 (0.0%)** | **1 / 50 (2.0%)** | **1** | **758.14** | **0.155%** | **41 / 150 (27.3%)** | **47 / 50 (94.0%)** | **88 / 200 (44.0%)** |
| **Train** | **All Combined** | 100 | **0 / 100 (0.0%)** | **1 / 100 (1.0%)** | **1** | **884.00** | **0.096%** | **41 / 200 (20.5%)** | **95 / 100 (95.0%)** | **136 / 300 (45.3%)** |
| **Val** | **Stage A** (Single) | 15 | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **0** | **1,091.87** | **0.025%** | **0 / 15 (0.0%)** | **11 / 15 (73.3%)** | **11 / 30 (36.7%)** |
| **Val** | **Stage B** (Multi) | 15 | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | **0** | **711.20** | **0.097%** | **11 / 46 (23.9%)** | **12 / 15 (80.0%)** | **23 / 61 (37.7%)** |
| **Val** | **All Combined** | 30 | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | **0** | **901.53** | **0.061%** | **11 / 61 (18.0%)** | **23 / 30 (76.7%)** | **34 / 91 (37.4%)** |
| **Test** | **Stage A** (Single) | 25 | **0 / 25 (0.0%)** | **0 / 25 (0.0%)** | **0** | **1,005.52** | **0.035%** | **0 / 25 (0.0%)** | **16 / 25 (64.0%)** | **16 / 50 (32.0%)** |
| **Test** | **Stage B** (Multi) | 25 | **0 / 25 (0.0%)** | **0 / 25 (0.0%)** | **0** | **617.04** | **0.174%** | **12 / 72 (16.7%)** | **22 / 25 (88.0%)** | **34 / 97 (35.1%)** |
| **Test** | **All Combined** | 50 | **0 / 50 (0.0%)** | **0 / 50 (0.0%)** | **0** | **811.28** | **0.105%** | **12 / 97 (12.4%)** | **38 / 50 (76.0%)** | **50 / 147 (34.0%)** |

---

## 4. Code Audit: Exact Source Lines Responsible for the Ambiguity

In `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/evaluate_exp06.py`:

```python
# Line 121:
resp_start_idx = len(p_toks)

# Line 131-136:
for i in range(resp_start_idx, len(tf_input)):
    tgt = tf_target[i]
    pr = int(torch.argmax(logits[i, :]).item())
    l = loss_fn(logits[i:i+1, :], torch.tensor([tgt])).item()
    resp_losses.append(l)
    resp_matches.append(pr == tgt)

# Line 146-147:
stage_a_tf_matches += sum(resp_matches)
stage_a_tf_tokens += len(resp_matches)
```

- **Analysis**:
  - `resp_matches` accumulated predictions over `range(resp_start_idx, len(tf_input))`.
  - In Stage A, `len(tf_input) == resp_start_idx + 2`.
  - Index `resp_start_idx` was the target value token ($T_1$).
  - Index `resp_start_idx + 1` was the `<EOS>` token ($1$).
  - Because the model easily learned that sequence length is 1 and predicted `<EOS>` after receiving $T_1$, Position 1 achieved **96% accuracy**.
  - Position 0 achieved **0% accuracy**.
  - Summing them together produced $(0 + 48) / 100 = 48.0\%$.

---

## 5. Audit Conclusions & Impact on Experiment 06 Findings

1. **Did the metrics come from the same examples, checkpoint, and positions?**
   - **YES**. All metrics used the exact same 180 records, the exact same checkpoint (`exp06_best_checkpoint.pt`), and the exact same causal logit tensor.
2. **Does the 48% accuracy mean the model learned to predict the target token?**
   - **NO**. The accuracy on predicting the target value token from context was **0.0%** (0 / 50 on Stage A train; 0 / 25 on Stage A test). The 48% was entirely driven by the model predicting `<EOS>` once the ground-truth target token was fed to it via teacher forcing.
3. **Does this change the central conclusion of Experiment 06?**
   - **NO, IT STRENGTHENS IT.**
   - Experiment 06 concluded: `MINIMAL COPYING NOT LEARNED`.
   - The original report noted a seemingly confusing paradox: "why is teacher-forced accuracy ~48% if the first token rank is 1,215 and probability is 0.04%?"
   - This audit proves there was **no paradox**: true teacher-forced target-value accuracy was **0.0%**, perfectly matching the rank (>1,000) and probability (0.04%).
   - The model completely failed to copy even single clean dictionary tokens from its prompt context.

---

## 6. Final Status & Recommendation

### Status:
`METRICS RECONCILED — ORIGINAL CONCLUSION SUPPORTED`

### Supported Conclusion:
The 3.44M-parameter ChakrMicro core cannot learn context-grounded string copying through unaugmented causal autoregressive next-token SFT. Neither JSON syntax overhead nor multi-token chunking was responsible for previous failures; the model fails to predict the target token even in a single-token, zero-delimiter template ($P \approx 0.04\%$, rank $> 1,000$, accuracy $0.0\%$).

### Single Recommended Next Step:
Transition the project away from autoregressive prompt-tuning for factual copying, and implement **explicit architectural induction** (e.g., evaluating or integrating the existing neural relation pointer head in `chakrview/cognition/pointer_integration.py` or constrained extractive token masking during runtime decoding).
