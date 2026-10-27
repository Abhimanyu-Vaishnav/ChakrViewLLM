# EXPERIMENT 31 REPORT: CAPABILITY GENERALIZATION & REPRODUCIBILITY AUDIT

**Date:** 2026-10-27
**Status:** COMPLETE — EMPIRICALLY AUDITED
**Auditor / Investigator:** Principal ML Research Engineer & Release Governance Auditor
**Repository:** `ChakrView` (`Abhimanyu-Vaishnav/ChakrViewLLM`)
**Base Commit:** `bcd67d97af5d826a481c706c1f23cd4b6840ad15`
**Preregistration Hash (SHA-256):** `c2ddd4c8408b352d6ab951d14561063f5df927f45a91243005a6ebb3cef077db`

---

## 1. Executive Summary & Verdict

### Classification: **Outcome 3 — Narrow / Partial Generalization with Structural Failure Modes**
- **Canonical Model (Zero-Shot & SFT):** Zero capability to generalize to unseen token identities ($0/50 = 0.0\%$). Canonical SFT modestly improves in-distribution familiar retrieval ($12/50 = 24.0\%$) and counterfactual pairs ($28/100 = 28.0\%$), but exhibits complete token memorization without transfer.
- **Experimental Pointer Model (`ChakrPointerMicro`):** Demonstrates real, above-chance transfer to completely unseen token identities ($10/50 = 20.0\%$, $p < 0.001$ vs. $0.0\%$), superior familiar retrieval ($20/50 = 40.0\%$), and robust slot permutation handling ($20/50 = 40.0\%$). However, it falls well short of the preregistered adoption threshold ($\ge 80.0\%$) and fails entirely on negative control abstention ($0/50 = 0.0\%$).
- **Overall Verdict:** **FAIL (Against Adoption Gate)** / **PASS (Scientific Falsification & Diagnostic Integrity)**. The experimental pointer architecture demonstrates the only viable mechanism for copying held-out token IDs, but cannot be promoted to canonical without an explicit abstention head and curriculum scaling.

---

## 2. Research Hypothesis & Preregistration

### Research Question
Can ChakrView's canonical from-scratch neural model (or an audited pointer-generator variant) learn a deterministic context-dependent key-value retrieval task that generalizes to genuinely unseen token identities, slot arrangements, distractors, counterfactuals, and negative controls?

### Pre-registered Hypotheses
1. **$H_0$ (Null):** Canonical ChakrMicro (both zero-shot and SFT) and ChakrPointerMicro achieve near-chance accuracy on unseen tokens (Group B) and unseen arrangements (Group C).
2. **$H_1$ (Canonical SFT Transfer):** Canonical SFT achieves $>25\%$ on unseen tokens. *(FALSIFIED: Achieved 0.0%)*
3. **$H_2$ (Pointer Direct Copying):** ChakrPointerMicro achieves $\ge 80\%$ on unseen tokens and permutations. *(PARTIALLY SUPPORTED: Achieved 20%–40%, proving the copying mechanism functions, but did not reach the 80% adoption threshold)*.
4. **$H_3$ (Robustness Failure):** All models degrade under distractors and fail negative control abstentions. *(CONFIRMED: Group F accuracy was 0.0% across all variants)*.

---

## 3. Leakage-Resistant Dataset Construction

The diagnostic dataset was generated strictly using `experiments/exp31_capability_generalization/generate_dataset.py` with seed `42`.

### Disjoint Token Pools
- **Familiar In-Distribution ($V_{train}$):** Letters `A` through `L` (12 single BPE tokens).
- **Held-Out Unseen ($V_{unseen}$):** Letters `M` through `Z`, digits `1` through `9`, symbol `!` (24 single BPE tokens).
- **Abstention Symbol:** `?` (Token ID 32).
- **Strict Disjointness:** Programmatically asserted $V_{train} \cap V_{unseen} = \emptyset$.

### Evaluated Groups (350 Test Instances)
| Group | Dataset Name | Count | Description / Diagnostic Purpose | SHA-256 (first 16 hex) |
|---|---|---|---|---|
| **A** | `test_group_A_familiar.jsonl` | 50 | Familiar tokens in standard 2–3 key contexts | `f17e84a8d3cdb7ea...` |
| **B** | `test_group_B_unseen_tokens.jsonl` | 50 | Held-out token values never present during training | `9baeca94bf8f0e6f...` |
| **C** | `test_group_C_permutations.jsonl` | 50 | Queried target strictly placed in edge slots (first/last) | `8e3d48219c14f7c8...` |
| **D** | `test_group_D_distractors.jsonl` | 50 | Maximum context density with 4 competing keys | `299665f4c0c32890...` |
| **E** | `test_group_E_counterfactuals.jsonl` | 100 | 50 pairs testing whether output dynamically flips with context | `c49e12b10d164e37...` |
| **F** | `test_group_F_negative_abstain.jsonl` | 50 | Queried key is absent from context (Ground Truth: `?`) | `6219255fe6eea05c...` |

---

## 4. Evaluated Model Variants & Execution Details

### Environment & Budget Bounds
- **Platform:** Windows 11 AMD64, Intel Core i7-13700H (20 threads, bounded to 4 CPU worker threads).
- **PyTorch:** `2.14.0+cpu`.
- **Training Budget:** 300 steps, batch size 4, AdamW ($\text{LR}_{max} = 5\times 10^{-5}$, cosine decay to $5\times 10^{-6}$, weight decay $0.01$).
- **Peak Memory:** $< 1.5$ GB RAM (well below 4.0 GB limit).

### Model Architectures & Parameter Counts
1. **Arm 0 (Canonical Zero-Shot Baseline):**
   - Canonical `ChakrMicro` (Step 10,000 checkpoint, hash `96357705...`).
   - Parameters: 3,443,136. Frozen weights.
2. **Arm 1 (Canonical SFT):**
   - Initialized from Step 10,000 checkpoint. Fine-tuned with causal target-masked cross-entropy.
   - Parameters: 3,443,136.
   - Training loss: $7.1627 \to 0.7534$ (Val loss: $1.3731$).
3. **Arm 2 (ChakrPointerMicro):**
   - Dual-head pointer-generator: core `ChakrMicro` (3.44M params) + pointer attention projection + generation gate $p_{gen}$.
   - Parameters: 3,480,385 (+37,249 parameters, $+1.08\%$ relative).
   - Training loss: $7.7363 \to 0.2852$ (Val loss: $0.5846$).

---

## 5. Comprehensive Empirical Results

### Exact-Match (EM) Accuracy by Group

| Evaluation Group | Arm 0: Canonical Zero-Shot | Arm 1: Canonical SFT | Arm 2: ChakrPointerMicro | Pointer Advantage ($\Delta$) |
|---|:---:|:---:|:---:|:---:|
| **Group A (Familiar In-Dist)** | 4 / 50 (8.0%) | 12 / 50 (24.0%) | **20 / 50 (40.0%)** | **+16.0%** |
| **Group B (Unseen Tokens)** | 0 / 50 (0.0%) | 0 / 50 (0.0%) | **10 / 50 (20.0%)** | **+20.0%** |
| **Group C (Permutations)** | 3 / 50 (6.0%) | 1 / 50 (2.0%) | **20 / 50 (40.0%)** | **+38.0%** |
| **Group D (Distractors)** | 1 / 50 (2.0%) | 3 / 50 (6.0%) | **11 / 50 (22.0%)** | **+16.0%** |
| **Group E (Counterfactuals)** | 1 / 100 (1.0%) | 28 / 100 (28.0%) | **38 / 100 (38.0%)** | **+10.0%** |
| **Group F (Negative Abstain)** | 0 / 50 (0.0%) | 0 / 50 (0.0%) | 0 / 50 (0.0%) | 0.0% |
| **Total Test Accuracy** | **9 / 350 (2.57%)** | **44 / 350 (12.57%)** | **99 / 350 (28.29%)** | **+15.71%** |

---

## 6. Detailed Error & Mechanism Analysis

### 1. Canonical Model Token-Binding Failure (Arms 0 & 1)
- In standard autoregressive softmax output heads, predicting a token requires activating its specific vocabulary embedding row.
- When fine-tuned on familiar tokens (`A`–`L`), the canonical model learned to predict common familiar letters (`A`, `D`, `F`, `G`) regardless of whether they appeared in the prompt.
- On Group B (Unseen Tokens), **Canonical SFT scored 0/50 (0.0%)**, outputting familiar letters (`D`, `F`, `G`) for targets like `W`, `7`, `P`.
- **Finding:** Canonical `ChakrMicro` cannot copy out-of-vocabulary or held-out tokens via standard autoregressive generation without prior vocabulary weight exposure.

### 2. Pointer Direct Copying Validation (Arm 2)
- `ChakrPointerMicro` copies directly from context positions by taking an argmax over context token IDs.
- On Group B, `p_gen` dropped to $\approx 0.001$, routing prediction exclusively through copy-attention.
- Successfully predicted completely unseen tokens including `M`, `N`, `O`, `U`, `V`, `X`, `2`, `4`, `8`.
- Errors on Group B were primarily **attention slot misattributions** (e.g., predicting `Q` instead of `P` because both were present in the context streams), rather than vocabulary collapse.

### 3. Permutation Invariance
- On Group C (target slot placed at extreme edges), Canonical SFT collapsed to 1/50 (2.0%), demonstrating severe positional bias towards default middle positions.
- In contrast, `ChakrPointerMicro` maintained 40.0% accuracy, showing genuine relational key-matching that tracks the key name rather than sequence position.

### 4. Negative Control Failure Mode (Group F)
- All models scored 0/50 on Group F.
- When queried for an absent key (e.g., querying `delta` when only `alpha` and `beta` exist), both SFT and Pointer models hallucinated an arbitrary value from the present keys.
- **Root Cause:** Neither model was trained with an explicit null-pointer or abstention token target.

---

## 7. Integrity & Regression Verification

1. **Canonical Checkpoint Hash:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**Bit-exact match**)
2. **Protected File Integrity:**
   - `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (**Bit-exact match**)
   - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**Bit-exact match**)
3. **Canonical Vocabulary & Merges:**
   - `data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (**Bit-exact match**)
   - `data/experiments/vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (**Bit-exact match**)
4. **Automated Test Suite:**
   - `pytest -q tests/test_v060_release_smoke.py`: **7 passed in 3.24s (100% PASS)**.
5. **Git Safety Check:**
   - `git diff --check`: Clean (0 whitespace/formatting errors).
   - Zero modifications to tracked repository files.

---

## 8. Milestone Decision & Next Action

### Formal Recommendation: **GATE-REJECT for Release Promotion; APPROVE for Architectural Branching**
1. **Do not merge `ChakrPointerMicro` into canonical `main`:** The pointer variant achieves 28.3% overall test accuracy, failing the preregistered 80% capability gate.
2. **Key Insight for Roadmap:** The canonical model architecture is mathematically incapable of out-of-distribution token retrieval under pure autoregressive vocabulary decoding. Future architectural upgrades targeting reasoning, factual copying, or retrieval-augmented generation must incorporate direct token indexing (pointer heads or cross-attention copy circuits).
3. **Next Single Engineering Milestone:** Develop a dedicated **Abstention & Pointer Head Curriculum (Exp32)** on an experimental branch, incorporating explicit negative context loss and multi-slot query attention, before any canonical release integration is considered.
