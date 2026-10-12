# Batch 79 — Previous Result Audit and Strict Benchmark Specification

## 1. Audit of Batches 75–78 Claims & Findings

The reported claims from Sprint Batches 75–78 were thoroughly audited against the local repository:

1. **Commit and Head Invariance**:
   - Starting HEAD `f5a72b325e933c221e8cbd9e20a7ac05ddd07e0d` was verified as the parent of `03de18f42242db5a51dfdae7cdd7397d3a746d17`.
   - The current branch is `dev/v0.6-v0.7-validation`.
2. **Canonical Architectural & Checkpoint Hashes**:
   - Parameter count: Exactly **3,443,136** parameters.
   - Fresh initialization SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
   - Step 10,000 checkpoint SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
3. **Modal Collapse Confirmation**:
   - In Batches 75–78, Condition B (synthetic pretraining) recorded 2/10 validation and 2/10 held-out test exact match.
   - Audit confirmed that **100% of all generated responses across all evaluation prompts were the constant digit `6`**.
   - The 2/10 validation and 2/10 test scores were exclusively artifacts of `6` matching pairs whose true sum was 6 (`4+2=6`, `6+0=6`, `0+6=6`, `1+5=6`).
4. **Token Budget & SFT Control Verification**:
   - Both Condition A and Condition B trained on exactly 56,231 tokens from their respective corpora for 100 steps.
   - User file `chakrview/runtime/session.py` remains uncommitted and untouched.
   - All 106 historical tests passed cleanly.

---

## 2. Full Addition Domain Benchmark Specification (Sums 0–18)

To definitively test whether a balanced curriculum can overcome modal collapse, Batch 79 introduces a full-domain addition benchmark:
- **Operand Domain**: $a, b \in \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$.
- **Sum Range**: $c = a + b \in [0, 18]$ (19 distinct sum classes).
- **Total Ordered Pairs**: Exactly 100 pairs.
- **Unordered Pairs**: Exactly 55 pairs $(u = (\min(a, b), \max(a, b)))$.

### 2.1 Unordered Grouping & Zero Reversed-Pair Leakage
To prevent the model from memorizing $(a, b)$ and trivially evaluating on $(b, a)$, **every pair $(a, b)$ and its reverse $(b, a)$ are assigned together to the same split**:
- **Train Set**: 35 unordered pairs $\to$ **64 ordered pairs** (covers **all 19 sum classes 0 through 18**).
- **Validation Set**: 10 unordered pairs $\to$ **18 ordered pairs**.
- **Held-Out Test Set**: 10 unordered pairs $\to$ **18 ordered pairs**.
- **Symmetry Diagnostic Set**: 32 off-diagonal ordered pairs from Validation and Test.
- **Template Diagnostic Set**: 36 test records phrased under alternate query/word templates.

### 2.2 Split Overlap Audit
- $\text{Train} \cap \text{Val}$ (ordered): **0**
- $\text{Train} \cap \text{Test}$ (ordered): **0**
- $\text{Val} \cap \text{Test}$ (ordered): **0**
- $\text{Train} \cap \text{Val}$ (unordered): **0**
- $\text{Train} \cap \text{Test}$ (unordered): **0**
- $\text{Val} \cap \text{Test}$ (unordered): **0**
- **Reversed-Pair Leakage**: **0 pairs** across all splits.

### 2.3 Constant-Output Baselines
In the training split, the most frequent answer sum is `6` (7 out of 64 pairs = 10.94%).
- Constant `6` predictor on Train: **10.94%** (7 / 64)
- Constant `6` predictor on Validation: **0.00%** (0 / 18, sum 6 is not in val)
- Constant `6` predictor on Held-Out Test: **0.00%** (0 / 18, sum 6 is not in test)

**Critical Generalization Rule**: If a model collapses to `6`, its validation and test exact-match score will be **0.0%**. A model can only score $>0\%$ on the held-out test set if it produces non-constant, conditioned predictions.

---

## 3. Tokenizer Encoding Properties for Sums 0–18

Unlike the $0..9$ domain where every answer was 1 token, sums $10..18$ have subtle BPE properties in the 4096-token vocabulary:
- Sums 0–9: 1 token each (`0` $\to 51, \dots, `9` \to 60$).
- Sums 10–16: 1 token each (`10` $\to 545$, `11` $\to 3655$, `12` $\to 466$, `13` $\to 1105$, `14` $\to 819$, `15` $\to 697$, `16` $\to 891$).
- Sum 17: **2 tokens** (`1` $\to 52$, `7` $\to 58$).
- Sum 18: 1 token (`18` $\to 1293$).

Our answer evaluator decodes greedy generation up to `<|im_end|>` and compares `clean_generated == str(expected_sum)`. This handles multi-token answers (`17`) without BPE misalignment.
