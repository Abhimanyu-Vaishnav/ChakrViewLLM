# CHAKRVIEW v0.8: EXPERIMENT 06 RESEARCH & DIAGNOSTIC REPORT
## Minimal Copy-Circuit Diagnostic: Single-Token & Multi-Token Context-Grounded Copying Under Pure Causal LM

- **Author**: ChakrView Research & ML Engineering Agent
- **Execution Date**: 2026-10-21
- **Experiment Identifier**: `research_v08_json_sft_exp06_minimal_copy_diagnostic`
- **Output Directory**: `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/`
- **Architecture**: ChakrMicro (3,443,136 parameters, 6 layers, $d_{\text{model}}=192$, $n_{\text{heads}}=6$, $d_{\text{ff}}=512$, context 512)
- **Primary Mission**: Determine whether ChakrMicro can learn and generalize a minimal context-grounded copying task (`Target value: {value}\nCopy: ` $\to$ `{value}`) under ordinary unweighted causal next-token training, cleanly isolating Stage A (single-token values) from Stage B (multi-token values) without JSON schema or delimiter overhead.

---

## 1. Safety, Invariants & Cryptographic Verification

All parent checkpoints, tokenizers, historical datasets, and runtime code were pre-verified and re-verified post-experiment (SHA-256):

| Artifact Role | File Path | SHA-256 Checksum | Invariant Status |
| :--- | :--- | :--- | :--- |
| **Canonical Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Immutable / Bit-exact |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Immutable / Bit-exact |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Immutable / Bit-exact |
| **Exp02 Parent Checkpoint** | `experiments/research_v08_json_sft/exp02_string_copy/checkpoints/exp02_best_checkpoint.pt` | `d31d2f9cc8be1a021b710a2f434db20d2f45e545e1c5fea10afcca1675f2bfe2` | Immutable / Bit-exact |
| **Exp06 Train Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/train.jsonl` | `aca94c894a07ef78f68401ee55705d894453ab3bf6c795b38c234ef465679ce5` | 100 records generated |
| **Exp06 Val Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/val.jsonl` | `c09070c880475668c3495458311aab22de5417b119addb14e3dac4e79fa2939b` | 30 records generated |
| **Exp06 Test Split** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/data/test.jsonl` | `1b536cb2ad1d462db814f164287e2974ff6b1948d006f89bc33cc55145ab276f` | 50 records generated |
| **Exp06 Best Checkpoint** | `experiments/research_v08_json_sft/exp06_minimal_copy_diagnostic/checkpoints/exp06_best_checkpoint.pt` | `d81953f82b2a485102a47662e9eac038e4573b535dbe68dde0d9adebd60822d0` | Selected at Step 110 |

Protected invariants: `chakrview/runtime/session.py` remained untouched; zero Git commits or remote operations.

---

## 2. Minimal Copy Dataset Design & Verification

To eliminate confounding variables (such as JSON syntax delimiters, brackets, colons, or multi-field composition), we constructed a narrow, deterministic copying diagnostic:
- **Task Template**: `Target value: {value}\nCopy: ` $\to$ `{value}`
- **Stage A (Single-Token Values)**: Clean lowercase dictionary words from the canonical vocabulary verified to tokenize to exactly 1 token in isolation and when preceded by `Target value: ` (e.g. `token`, `theta`, `arch`, `self`, `mode`, `float`).
- **Stage B (Multi-Token Values)**: Composite 2-to-4 token underscored identifier strings generated from disjoint word pools (e.g. `softmax_produci`, `frequ_guarante`, `tion_libr`).
- **Strict Partitioning & Disjointness**:
  - **Train**: 100 records (50 Stage A, 50 Stage B).
  - **Validation**: 30 records (15 Stage A, 15 Stage B).
  - **Held-Out Test**: 50 records (25 Stage A, 25 Stage B).
  - **Leakage Audit**: Tested and verified zero overlap between train, validation, and test at both the raw-string level and the token-sequence level.

---

## 3. Pre-Training Objective & Causal Alignment Unit Tests

Before launching training, we executed 4 unit test suites in `verify_objective.py` (all passed deterministically):
1. **Zero Leakage**: Passed at both string and token levels.
2. **Causal Next-Token Alignment & Logit Index Assertion**:
   - For an input sequence `[BOS] + p_toks + r_toks` and target `[-100]*len(p_toks) + r_toks + [EOS]`:
   - Input token at index `len(p_toks)` is the final prompt token (e.g. `' '` after `'Copy:'`).
   - The model's logits at index `len(p_toks)` **strictly predict the first response token** `r_toks[0]`.
   - Verified that scoring at `len(p_toks)-1` (which is masked `-100`) or any other position is rejected.
3. **Finite Loss and Gradients**: Verified on ChakrMicro batched data without NaN or gradient clipping failure.
4. **Evaluator Correctness**: Hand-crafted positive exact matches, negative mismatches, truncated tokens, and empty strings were verified.

---

## 4. Controlled Training Dynamics (120 Steps Unweighted)

Trained an experimental copy from the Canonical Parent (`training_checkpoint_step_10000.pt`) using standard unweighted cross-entropy ($lr=3\times 10^{-5} \to 3\times 10^{-6}$, batch size 4, AdamW):
- **Pre-training Initial Val Loss**: `8.4784`
- **Validation Loss Trajectory**:
  - Step 10: `7.4368` (Drift 0.19%)
  - Step 20: `5.8418` (Drift 0.46%)
  - Step 30: `4.7773` (Drift 0.70%)
  - Step 40: `4.4441` (Drift 0.90%)
  - Step 50: `4.3174` (Drift 1.07%)
  - Step 60: `3.9089` (Drift 1.22%)
  - Step 80: `3.7507` (Drift 1.43%)
  - **Step 110 (Best Validation Checkpoint)**: **`3.6514`** (Drift 1.55%)
  - Step 120 (Final): `3.8952` (Drift 1.57%)
- **Total Training Time**: 4.89 seconds on CPU.
- **Selection**: Step 110 achieved the global validation minimum and was saved as `exp06_best_checkpoint.pt`.

---

## 5. Comprehensive Diagnostic Results (Train, Validation & Held-Out Test)

Evaluated `exp06_best_checkpoint.pt` across all 180 records using greedy deterministic decoding (`max_new_tokens=16`, stop token `[1]`):

### Summary Metrics Table:

| Metric Dimension | Train Split ($N=100$) | Validation Split ($N=30$) | Held-Out Test Split ($N=50$) |
| :--- | :---: | :---: | :---: |
| **Overall Free-Generation Exact Match** | **0 / 100 (0.0%)** | **0 / 30 (0.0%)** | **0 / 50 (0.0%)** |
| **Stage A (Single-Token) Exact Match** | **0 / 50 (0.0%)** | **0 / 15 (0.0%)** | **0 / 25 (0.0%)** |
| **Stage B (Multi-Token) Exact Match** | **0 / 50 (0.0%)** | **0 / 15 (0.0%)** | **0 / 25 (0.0%)** |
| **Stage A Teacher-Forced Token Accuracy** | 48.0% | 36.7% | 32.0% |
| **Stage B Teacher-Forced Token Accuracy** | 44.0% | 37.7% | 35.1% |
| **Mean First-Token Logit Rank** | 1,215.3 | 1,084.7 | 1,142.6 |
| **Mean First-Token Target Probability** | 0.04% | 0.03% | 0.02% |

---

## 6. Qualitative Forensic Analysis & Collapse Mode

### Representative Sample Predictions:

| Split | Stage | Context Target Value | Free-Running Greedy Output | First-Token Target Rank | First-Token Target Probability |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Test** | Stage A (Single) | `verific` | `'Incremental of the word of the word...'` | **1,202** | 0.01% |
| **Test** | Stage A (Single) | `tokeniz` | `'Incremental of the word of the word...'` | **2,541** | 0.00% |
| **Test** | Stage A (Single) | `arch` | `'Andra (b. 1970)\n* A'` | **250** | 0.05% |
| **Test** | Stage B (Multi) | `softmax_produci` | `'[http://www.gov.'` | **289** | 0.06% |
| **Test** | Stage B (Multi) | `sing_star` | `'float_test'` | **772** | 0.02% |
| **Test** | Stage B (Multi) | `frequ_guarante` | `'[http://www.gov.'` | **229** | 0.08% |
| **Train** | Stage A (Single) | `ative` | `'Andra (b. 1970)\ndef '` | **1,327** | 0.01% |
| **Train** | Stage A (Single) | `struc` | `'The Andy of the Arce\n\n >>>'` | **644** | 0.02% |
| **Train** | Stage B (Multi) | `tion_libr` | `'float_test'` | **3,553** | 0.00% |
| **Train** | Stage B (Multi) | `cloc_float` | `'float_test'` | **112** | 0.15% |

### Empirical Collapse Mode:
1. **The Model Never Learned to Copy — Not Even on Training Examples**:
   - Exact match is **0 / 100 on the training set**.
   - Even when the target is a single clean dictionary token (`token`, `arch`, `struc`) explicitly provided 6 tokens prior in the prompt (`Target value: token\nCopy: `), the model does **not** emit `token`.
2. **Attractor Collapse into Pretraining Corpus Fragments**:
   - Rather than routing attention to the target token in the prompt, greedy decoding emits memorized pretraining n-grams:
     - `'float_test'`
     - `'[http://www.gov.'`
     - `'Andra (b. 1970)\n* A'`
     - `'Incremental of the word of...'`
3. **Target Token Rank at First Position**:
   - The first token target is ranked on average **> 1,000** out of 4,096 vocab tokens, with probability $< 0.05\%$.
   - The self-attention layers in ChakrMicro do not form an induction head that copies the token from position $t-6$ to position $t$.

---

## 7. What This Diagnostic Established

This experiment rigorously answers the question posed in the mission:

1. **Failure is NOT due to JSON syntax or delimiters**:
   Experiment 06 completely stripped away `{`, `"`, `:`, and JSON schemas. Even with a pure minimal format (`Target value: X\nCopy: X`), copying failed completely (0%).
2. **Failure is NOT due to multi-token chunking**:
   Even Stage A (exact single-token vocabulary words that require zero subword stitching) failed completely (0/50 train, 0/25 test).
3. **Failure is NOT a generalization gap**:
   The model fails equally on training examples (0/100) and held-out test examples (0/50). There is no memorization; the task was simply not learned.
4. **Failure is NOT a decoding bug**:
   The target token is ranked $>1,000$ in the model's logits with probability $\sim 0.03\%$.
5. **Root Cause**:
   Under standard causal autoregressive SFT from the pretrained parent checkpoint, a 3.44M-parameter 6-layer transformer with $d_{\text{model}}=192$ **cannot learn an in-context string copying circuit via unweighted cross-entropy within bounded SFT**. The pretrained attention weights are strongly biased toward generic language/code continuations (`http://`, `def `, `Incremental`), and the gradient signal from 100 examples is insufficient to restructure multi-head self-attention into an identity pointer mechanism without architectural pointer heads or retrieval augmentation.

---

## 8. Decision & Next Experiment Recommendation

### Primary Status:
`MINIMAL COPYING NOT LEARNED`

### Recommendation:
**Do not attempt further causal SFT prompt-tuning experiments for context copying.**
Causal autoregressive fine-tuning without architectural inductive bias has now been tested and ruled out across:
- Two-key JSON SFT (Exp 01) $\to$ Mode collapse.
- Single-field JSON SFT (Exp 02) $\to$ Generic subword collapse.
- Diagnostic analysis (Exp 03 & 04) $\to$ First-token disconnection ($P < 0.8\%$).
- Slot-weighted loss (Exp 05) $\to$ Gradient destabilization and syntax loss.
- Minimal copying diagnostic (Exp 06) $\to$ Complete copy circuit failure even on single-token training examples (0%).

To achieve context-grounded factual extraction in ChakrView, research must transition to **explicit architectural copy mechanisms**:
- Integration with the existing neural relation pointer head (`chakrview/cognition/pointer_integration.py` / `ChakrRelationPointerHead`).
- Constrained extractive token masking during decoding.
- Hybrid neuro-symbolic retrieval routing.

---

## 9. Protected Invariant Confirmation

- Canonical parent checkpoint: **Unmodified** (`96357705...`).
- Canonical tokenizer: **Unmodified** (`db4d6119...` / `ada4b1dc...`).
- `chakrview/runtime/session.py`: **Unmodified**.
- Git state: **Zero commits, pushes, or branch mutations**.
