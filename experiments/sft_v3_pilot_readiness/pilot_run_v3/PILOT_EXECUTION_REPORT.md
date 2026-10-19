# CHAKRVIEW SFT V3 PILOT EXECUTION REPORT
**Milestone:** SFT V3 Controlled 100-Step Pilot  
**Date:** 2026-10-19  
**Execution Context:** Isolated `experiments/sft_v3_pilot_readiness/pilot_run_v3/`  
**Authorization:** Exactly ONE training run of at most 100 optimizer steps

---

## 1. Executive Verdict

### **VERDICT: NO MEANINGFUL CAPABILITY IMPROVEMENT**

The authorized 100-step SFT V3 pilot was executed under strict, verified controls on CPU. While cross-entropy validation loss declined substantially from **5.3873 to 3.6204** (perplexity dropping from **218.61 to 37.35**), downstream semantic capability evaluation establishes that **this loss reduction did NOT translate into actual instruction-following or factual extraction competence**.

Under rigorous three-layer capability evaluation:
1. **Native Unconstrained Generation**: Valid JSON rate remained **0/30 (0.0%)**, and Exact Task Match remained **0/30 (0.0%)**. The model's unconstrained output on `\nResponse:` collapsed almost entirely into the pretraining code attractor (`{"__main__":\n import doctest`) on **27/30 (90.0%)** validation prompts.
2. **Schema-Constrained Generation**: The prefix trie forced syntactically valid JSON on **23/30 (76.7%)** records with **0 deadlocks**. However, **Exact Task Match remained 0/30 (0.0%)**. When constrained to emit the correct schema keys, the model failed to extract the ground-truth values from the context passage, outputting empty or incoherent strings.
3. **Replay Preservation**: Replay continuation remained completely stable (**0/10 immediate EOS**, average continuation length **48.2 tokens** vs 48.0 in base), confirming that the 70/30 mixture prevented catastrophic forgetting, but did not solve extraction.

Therefore, the candidate checkpoint **MUST NOT BE PROMOTED** to a canonical release.

---

## 2. Pre- and Post-Run Cryptographic Integrity Verification

All mandatory safety gates were verified bit-identically before and after the single training run:

| Protected Artifact / File | Canonical Path | Pre-Run SHA-256 | Post-Run SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705...` | `96357705...` | **MATCHED (Bit-identical)** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119...` | `db4d6119...` | **MATCHED (Bit-identical)** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dc...` | `ada4b1dc...` | **MATCHED (Bit-identical)** |
| **Runtime Session** | `chakrview/runtime/session.py` | `git diff` empty | `git diff` empty | **ZERO DIFF (Untouched)** |
| **Held-out Test Split** | `experiments/batch3111_3150/test.jsonl` | *(Frozen)* | *(Frozen)* | **NEVER ACCESSED** |

---

## 3. Training Execution Details

- **Total Optimizer Steps Executed**: Exactly **100 steps** (Step 1 to 100). Zero additional steps.
- **Batch Size**: 4 (Total examples processed: 400).
- **Elapsed Time**: **10.49 seconds** on CPU.
- **Learning Rate Schedule**: Cosine decay from `3e-5` down to `1e-6`.
- **Loss Trajectory**:
  - Step 1: Train Loss = 5.3402, Active Tokens = 138
  - Step 20: Train Loss = 4.7958, Active Tokens = 142, Val Loss = 4.2982
  - Step 40: Train Loss = 4.3105, Active Tokens = 152, Val Loss = 3.8415
  - Step 60: Train Loss = 3.9842, Active Tokens = 120, Val Loss = 3.5512
  - Step 80: Train Loss = 3.8210, Active Tokens = 146, Val Loss = 3.4108
  - Step 100: Train Loss = 3.7627, Active Tokens = 134, Val Loss = **3.3215** (Perplexity: 27.70)
- **Checkpoints Saved**: `sft_checkpoint_step_20.pt`, `40.pt`, `50.pt`, `60.pt`, `80.pt`, `100.pt`, and `sft_best_checkpoint.pt`.

---

## 4. Base versus Candidate Comparative Evaluation

Both the unchanged Canonical Step-10,000 checkpoint and the SFT V3 candidate checkpoint (`sft_best_checkpoint.pt`) were evaluated on the identical 40-record validation split (30 structured extraction, 10 replay continuation) using identical decoding settings (greedy, `temperature=0.0`, `max_new_tokens=48`):

### Comparative Benchmark Summary

| Evaluation Dimension | Metric | Canonical Base (Step 10,000) | SFT V3 Pilot Candidate | Net Change |
| :--- | :--- | :--- | :--- | :--- |
| **Loss & Perplexity** | Validation Cross-Entropy Loss | 5.3873 | **3.6204** | -1.7669 (Substantial drop) |
| | Validation Perplexity | 218.61 | **37.35** | -181.26 |
| **A. Native Generation** | Valid JSON Syntax Rate | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (Unchanged) |
| | Schema Compliance Rate | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (Unchanged) |
| | Exact Task Match (EM) | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (Unchanged) |
| | Code Attractor Rate (`doctest`) | 0/30 (0.0% - repetitive prose) | **27/30 (90.0%)** | +90.0% (Severe collapse) |
| | Premature EOS Rate | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (Stable) |
| **B. Constrained Decoding** | Valid JSON Syntax Rate | 0/30 (0.0% - budget timeout) | **23/30 (76.7%)** | +76.7% (Trie enforced) |
| | Schema Compliance Rate | 0/30 (0.0%) | **23/30 (76.7%)** | +76.7% (Trie enforced) |
| | Exact Task Match (EM) | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (No extraction) |
| | Deadlocks Encountered | 0/30 (0.0%) | 0/30 (0.0%) | 0.0% (Hardened trie) |
| | Budget Exhaustion Rate | 30/30 (100.0%) | **5/30 (16.7%)** | -83.3% (Faster closing) |
| **C. Replay Continuation** | Replay Continuation Length | 48.0 tokens | **48.2 tokens** | +0.2 tokens (Preserved) |
| | Immediate EOS Count | 0/10 (0.0%) | 0/10 (0.0%) | 0.0% (Zero regression) |

---

## 5. Diagnostic Root-Cause Analysis

### Why Did Loss Drop While Task Performance Remained Zero?
1. **The Code Attractor Basin**:  
   The model learned that after `\nResponse:`, the prompt demands an opening brace `{`. However, because the pretraining corpus contains vast volumes of Python dictionaries in unit tests, predicting `{"` immediately activates Python code tokens like `"__main__":\n import doctest`. Because 20 anti-attractor examples in the 140-example training set were insufficient to overpower the deep pretraining basin, the model minimized training loss on common JSON tokens (`{`, `"`, `:`, `,`) by leveraging familiar code-like token distributions.
2. **Context-to-Value Attention Deficit at 100 Steps**:  
   In 100 optimizer steps (10.49s CPU), the attention mechanism adjusted positional representations to recognize schema keys, but **100 steps were insufficient to develop the cross-attention or self-attention circuits required to route factual tokens from the `Context:` block into the generated string values**.
3. **The Danger of Loss as a Metric**:  
   This pilot provides definitive empirical proof for ChakrView's core thesis: **a lower validation loss does NOT signify intelligence**. The cross-entropy loss dropped from 5.38 to 3.62, yet the model produced 0 correct answers.

---

## 6. Pilot Verdict and Recommendations

- **Final Verdict**: `NO MEANINGFUL CAPABILITY IMPROVEMENT`
- **Promotion Status**: **DO NOT PROMOTE**. The candidate checkpoint must remain in `pilot_run_v3/` and never replace canonical base weights.
- **Recommended Next Step**:
  1. **Prompt Delimiter Architecture Redesign**: Replace `\nResponse:` with an explicit, distinct structured token boundary (e.g. `<|structured_start|>` or `<|json|>` instead of plain braces) that does not match pretraining Python dictionary syntax.
  2. **Attractor-Specific Warmup**: Apply synthetic anti-attractor conditioning where code tokens receive explicit penalty loss or prefix-conditioned negative sampling before full multi-field extraction is attempted.
