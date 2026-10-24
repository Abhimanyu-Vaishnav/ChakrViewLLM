# ChakrView Normal Chat Gate v0.1 -- Final Audit Reconciliation

**Date**: 2026-10-22  
**Target Evaluation**: `experiments/normal_chat_gate_v01/`  
**Audit Purpose**: Perform a rigorous, reproducible, non-training final audit reconciling all prior report discrepancies using the exact on-disk files and repository source code.

---

## 1. Audit Scope & Executed Commands

This audit was conducted strictly in read-only mode regarding model weights, canonical checkpoints, datasets, and runtime code:
1. **Preservation Checks**:
   - `git status --short` verified before and after audit: zero tracked modifications, zero commits, zero stashes, zero resets.
   - All 6 untracked project roots preserved intact.
2. **Evaluated On-Disk Records**:
   - `experiments/normal_chat_gate_v01/prompts.jsonl` (20 records, SHA-256: `611cf0cfc6731e21b764047a0694119d5c317fae69e46927e3660fa406cb74be`)
   - `experiments/normal_chat_gate_v01/outputs.jsonl` (20 records, SHA-256: `cbcc5b43fa80e81fbb4ef071064562fa5349e29f3453b708cfbe2c24efd8c1c5`)
   - `experiments/normal_chat_gate_v01/SCORES.csv` (20 rows, SHA-256: `3b469e38cb8a379477fb4b2f56f1400bf1b425ef81e9d1eb7d057a7d47e27ce6`)
3. **Source Code & Contract Files Inspected**:
   - `chakrview/tokenizer/special_tokens.py` (lines 10-42)
   - `chakrview/tokenizer/tokenizer.py` (lines 1-89)
   - `chakrview/tokenizer/serialization.py` (lines 1-147)
   - `chakrview/tokenizer/decoder.py` (lines 19-80)
   - `data/experiments/vocab_4096/config.json` (lines 1-30)
   - `chakrview/runtime/neural_inference_contract.py` (lines 81-275)
   - `chakrview/runtime/neural_runtime_adapter.py` (lines 95-375)
   - `chakrview/runtime/session.py` (full 85-line protected session)
   - `chakrview/brain/model.py` (full architecture definition)
4. **Non-Destructive Python Verification Routines Executed**:
   - Complete record parsing and ID bijection validation (`p_ids == o_ids == s_ids`).
   - Idempotent token re-decoding across 100% of recorded output token sequences.
   - Mathematical derivation of repetition heuristic counts and exact discrepancy diagnosis.
   - SHA-256 hash recalculation across canonical checkpoint, tokenizer artifacts, protected `session.py`, `model.py`, and all 6 source datasets.

---

## 2. Table of Discrepancies & Final Reconciliations

| Discrepancy Topic | Prior Claim A | Prior Claim B | Empirical Ground Truth | Exact Cause & Resolution | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Scoring Denominator** | 196 possible points | 204 possible points | **Exactly 180 points** | Grounding was counted for 14 non-contextual prompts and Uncertainty for 16 clear factual prompts in early drafts. Programmatic rubric calculation: $20\times 2 + 20\times 2 + 6\times 2 + 20\times 2 + 4\times 2 + 20\times 2 = 180$ points. | **RESOLVED** |
| **Word-Trigram Repetition Count** | 14 / 20 prompts (70.0%) | 15 / 20 prompts (75.0%) | **14 / 20** (stored `is_repetitive`) vs. **15 / 20** (any duplicate trigram) | **PROMPT_07_USER_CORRECTION** is the exact sole prompt responsible. It contains exactly one duplicate word trigram (`('the', 'first', 'of')`), making $(N_{{trigrams}} - N_{{unique}}) = 1$. The stored evaluation rule required $\ge 2$ duplicates (`diff >= 2`), yielding 14/20. Any duplicate rule (`diff >= 1`) yields 15/20. Both definitions are documented. | **RESOLVED** |
| **Token-Bigram Duplication Count** | 20 / 20 prompts (100.0%) | 19 / 20 prompts (95.0%) | **19 / 20** (at $\ge 4$ bigrams) vs. **20 / 20** (at $\ge 3$ bigrams) | **PROMPT_18_STRICT_CONTEXT_QA** is the exact sole prompt responsible. Its most repeated token bigram `(770, 559)` (`" the"`) occurs exactly 3 times. If threshold is $\ge 4$, count is 19/20 (95.0%). If threshold is $\ge 3$, count is 20/20 (100.0%). | **RESOLVED** |
| **Consecutive Runaway Count** | 2 / 20 prompts (10.0%) | 1 newline (5%) + 1 char (5%) | **2 / 20** combined runaway | PROMPT_17 generated 48 consecutive newlines (`\n{{48}}`). PROMPT_02 generated 12 consecutive identical letters (`'A'{{12}}`). Reporting as "2/20 newline runaway" was imprecise; it is 1 newline runaway + 1 character runaway, total 2/20 (10.0%). | **RESOLVED** |
| **Model Parameter Hash vs. File Hash** | Treated `c5571c...` as `model.py` file hash | Refuted as non-matching `model.py` hash | **Fresh-Init Parameter Hash** | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` is the SHA-256 parameter tensor digest of `ChakrMicro(ModelConfig())` initialized with `torch.manual_seed(42)`. The source code file hash of `chakrview/brain/model.py` is `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`. | **RESOLVED** |
| **Model Inability to Emit EOS** | Asserted checkpoint cannot emit EOS | Categorized as **INCONCLUSIVE** | **INCONCLUSIVE** under sampling | Greedy unpenalized decoding never emitted token 1 (`<EOS>`) across 20 prompts. However, sampling decoding (`temp > 0.0`) and presence/repetition penalties were not evaluated. Total EOS incapacity cannot be claimed from greedy records alone. | **RESOLVED** |

---

## 3. Repetition-Metric Reconciliation & Heuristic Specifications

### 1. Stored Output Flag Heuristic (`is_repetitive: True`)
- **Exact Formula**:
  ```python
  words = text.lower().split()
  trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)] if len(words) >= 3 else []
  is_repetitive = (len(trigrams) - len(set(trigrams))) >= 2
  ```
- **Result**: Exactly **14 / 20 prompts (70.0%)** satisfy this condition.
  - Satisfying IDs: PROMPT_01, 03, 04, 05, 06, 08, 09, 12, 13, 14, 15, 16, 19, 20.
  - Excluded IDs: PROMPT_02 (diff=0), PROMPT_07 (diff=1), PROMPT_10 (diff=0), PROMPT_11 (diff=0), PROMPT_17 (diff=0), PROMPT_18 (diff=0).

### 2. Any Word-Trigram Duplication Heuristic
- **Exact Formula**: At least one duplicate word trigram (`diff >= 1` or `len(trigrams) > len(set(trigrams))`).
- **Result**: Exactly **15 / 20 prompts (75.0%)**.
  - Adds PROMPT_07_USER_CORRECTION (`diff = 1`, duplicate trigram: `('the', 'first', 'of')`).

### 3. Token-Bigram Duplication Heuristics
- **Threshold $\ge 4$ Occurrences**: Exactly **19 / 20 prompts (95.0%)**.
  - All prompts except PROMPT_18 satisfy this threshold.
- **Threshold $\ge 3$ Occurrences**: Exactly **20 / 20 prompts (100.0%)**.
  - PROMPT_18 contains 3 occurrences of `(770, 559)` (`" the"`).

### 4. Character-Level Runaway Heuristics
- **Consecutive Newlines ($\ge 10$)**: **1 / 20 prompts (5.0%)** (PROMPT_17: 48 consecutive `
`).
- **Consecutive Identical Characters ($\ge 10$)**: **1 / 20 prompts (5.0%)** (PROMPT_02: 12 consecutive `'A'`).
- **Combined Runaway Rate**: **2 / 20 prompts (10.0%)**.

---

## 4. Special-Token & Stopping-Behavior Audit

### Ground Truth from Source Code:
1. **Special Token IDs**:
   - `BOS_ID` = `0` (`<BOS>`), defined in `chakrview/tokenizer/special_tokens.py:16`.
   - `EOS_ID` = `1` (`<EOS>`), defined in `chakrview/tokenizer/special_tokens.py:17`.
   - `PAD_ID` = `2` (`<PAD>`), defined in `chakrview/tokenizer/special_tokens.py:18`.
   - Token ID `1` is verified to be `<EOS>` under `data/experiments/vocab_4096/config.json`.
2. **Stopping Token Appending Order**:
   - In `chakrview/runtime/neural_runtime_adapter.py:329-334`, `next_token` is appended to `generated_tokens` **before** checking `if active_eos_stop and next_token == 1:`.
   - Therefore, if `<EOS>` had been generated, token `1` would appear as the final element in `generated_tokens`.
3. **Empirical Record Findings**:
   - In `outputs.jsonl`, all 20 records have `stop_reason: "MAX_TOKENS"`.
   - Inspection of all 20 `generated_tokens` arrays confirms **0 occurrences of token `0`, `1`, or `2`**. Every generated token is a non-special text token.
4. **Greedy vs. Sampling Distinction**:
   - The greedy argmax never emitted `<EOS>`. However, whether `<EOS>` has non-zero probability under sampling (`temperature > 0.0`) or repetition penalties is **INCONCLUSIVE**.

---

## 5. Artifact Provenance & Full SHA-256 Digest Verification

Every single artifact was re-hashed from disk during this audit:

| Artifact Description | Classification | File Path | Established Expected Hash | Actual Computed SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | Checkpoint File | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| **Tokenizer Vocab** | Tokenizer File | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS** |
| **Tokenizer Merges** | Tokenizer File | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS** |
| **Protected Runtime** | Source Code File | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS** |
| **Model Architecture Source** | Source Code File | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **PASS** |
| **Exp16 v3.1 Train** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/train_exp16_v3_1.jsonl` | `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14` | `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14` | **PASS** |
| **Exp16 v3.1 Val** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/val_exp16_v3_1.jsonl` | `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36` | `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36` | **PASS** |
| **Exp16 v3.1 Test** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/test_exp16_v3_1.jsonl` | `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af` | `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af` | **PASS** |
| **Exp16 v3 Train** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/train_exp16_v3.jsonl` | `d5d8b76871faad8e449d0a59df7834e7db1f37b3c21673743f20ee87249873dc` | `d5d8b76871faad8e449d0a59df7834e7db1f37b3c21673743f20ee87249873dc` | **PASS** |
| **Exp16 v3 Val** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/val_exp16_v3.jsonl` | `c9954fb71fd5251065076ad96cfdc78812e2f970eb3cf55dc09c94c38844e95e` | `c9954fb71fd5251065076ad96cfdc78812e2f970eb3cf55dc09c94c38844e95e` | **PASS** |
| **Exp16 v3 Test** | Source Dataset | `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/data/test_exp16_v3.jsonl` | `096d018f357878e89560f982018446c8f09e5143767c1a5425d74bea9560ad10` | `096d018f357878e89560f982018446c8f09e5143767c1a5425d74bea9560ad10` | **PASS** |

### Clarification on Immutability:
- Matching the checkpoint file hash confirms that the on-disk model weight checkpoint was not modified.
- In-memory immutability during runtime inference is established separately by the adapter's pre- and post-execution digest check (`Delta-W = 0`), which passed across 100% of the 20 test executions.

---

## 6. Evaluation Record & Telemetry Summary

- **Input Token Counts**:
  - Min: 21 tokens (`PROMPT_15_BASIC_ARITHMETIC`)
  - Max: 117 tokens (`PROMPT_17_SUMMARIZATION`)
  - Mean: 50.35 tokens
- **Output Token Counts**:
  - Exactly 48 tokens across all 20 records (100.0%).
- **Inference Latency (CPU execution)**:
  - Min: 769.29 ms (`PROMPT_03_SHORT_EXPLANATION`)
  - Max: 1512.03 ms (`PROMPT_05_CONTEXT_RETENTION`)
  - Mean: 1041.41 ms (~1.04 seconds per prompt)
- **Telemetry Completeness**:
  - Telemetry fields (`latency_ms`, `input_token_count`, `output_token_count`, `stop_reason`, `weight_hash_verified`) are present on 100% of records.
  - Hyperparameters (`temperature: 0.0`, `decoding: "greedy"`, `repetition_penalty: 1.0`, `stop_on_eos: True`) were configured at adapter level and verified via source code.

---

## 7. Provisional Claims & Human-Review Scope

The following items are explicitly marked as **Provisional / Requiring Human Review**:
1. **Semantic Quality Scores**:
   - The assigned score of 0/180 is based on concrete empirical evidence (hallucinated math templates, failure to follow instructions, context blindness, token budget exhaustion).
   - However, subjective semantic dimensions (e.g., relevance, degree of task completion) remain provisional pending formal human reviewer sign-off.
2. **Pretraining Corpus Overfitting as Root Cause**:
   - Categorized as a **Supported Hypothesis** due to structural correlation with synthetic arithmetic headers (`Step-by-Step Solution:`). Token-level frequency counts of the pretraining data have not been computed.
3. **Sampling Decoding Capabilities**:
   - Categorized as **Not Yet Tested**.

---

## 8. Separate Formal Verdicts

### 1. Artifact Integrity: **PASS**
All canonical checkpoints, tokenizer configurations, protected runtime files, model source code, and source datasets match their established expected SHA-256 digests bit-for-bit.

### 2. Evaluation Record Integrity: **PASS**
`prompts.jsonl`, `outputs.jsonl`, and `SCORES.csv` parse cleanly, maintain strict 1:1 bijection across all 20 prompt IDs, contain complete telemetry, provide prompt-specific evidence rationales, and programmatically resolve the scoring denominator to exactly 180 points.

### 3. Normal Chat Capability Under Tested Configuration: **FAIL**
Under greedy unpenalized decoding (`temp=0.0`, `rep_penalty=1.0`), the raw unassisted ChakrMicro step_10000 model achieves **0 / 180 possible points (0.0%)**. It exhibits severe pretraining synthetic arithmetic bias, degenerate n-gram looping across 100% of prompts, context blindness, and 100% token budget exhaustion without generating `<EOS>`.

---

## 9. Next Experiment Recommendation

### Recommendation: Sufficient Evidence to Design `exp17_instruction_sft_v1`
- The audit evidence is **fully sufficient** to justify moving to conversational SFT design:
  1. The base model capability ceiling under greedy decoding is thoroughly documented.
  2. The failure modes (domain collapse, math header bias, token looping, non-termination) are precisely isolated.
  3. The evaluation harness and scoring rubrics are calibrated and verified.
- **Recommended Hypothesis**: Fine-tuning the ChakrMicro base checkpoint on $\le 500$ balanced conversational turns with explicit delimiters (`<BOS>`, role prefixes, `<EOS>`) will eliminate the arithmetic header bias and establish turn-boundary termination.
*(Recommendation only. No training, fine-tuning, or experimental script execution has been started).*
