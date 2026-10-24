# ChakrView Normal Chat Gate v0.1 -- Audit and Validation Corrections

## 1. Executive Summary of Corrections
This document provides an evidence-based audit, inventory validation, and reconciliation of the initial Normal Chat Gate v0.1 deliverables against the active repository state.

---

## 2. Inventory and Record Validation
- **Record Counts**:
  - prompts.jsonl: 20 records, 20 unique IDs, 0 duplicate IDs, 0 missing IDs.
  - outputs.jsonl: 20 records, 20 unique IDs, 0 duplicate IDs, 0 missing IDs.
  - SCORES.csv: 20 rows, 20 unique IDs, 0 duplicate IDs, 0 missing IDs.
  - Exact 1-to-1 matching: Every output matches exactly one prompt ID (p_ids == o_ids == s_ids).
- **Prompt and Context Formatting Agreement**:
  - Re-executing ormat_conditioned_prompt(prompt, context) across all 20 records produced 0 mismatches with the stored ormatted_prompt fields.
- **Output Record Field Completeness**:
  - 100% of output records contain:
    - Checkpoint identity: 	raining_checkpoint_step_10000.pt
    - Checkpoint SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
    - Token counts: input_token_count (18 to 110), output_token_count (48 for all 20)
    - Stop reason: MAX_TOKENS (all 20)
    - Latency: latency_ms (ranging from 873.3 ms to 1152.0 ms)
    - Weight verification: weight_hash_verified = True

---

## 3. Scoring Denominator and Metric Reconciliation
- **Previous Discrepancy**: Prior drafts contained conflicting denominator claims (196 vs 204 total possible points).
- **Programmatic Truth**:
  - Relevance: 20 prompts * 2 = 40 points (0 N/A)
  - Instruction Following: 20 prompts * 2 = 40 points (0 N/A)
  - Grounding in Context: 6 prompts * 2 = 12 points (14 N/A: PROMPT_01..04, 08, 09, 11..16, 19, 20)
  - Task Completion / Reasoning: 20 prompts * 2 = 40 points (0 N/A)
  - Appropriate Uncertainty: 4 prompts * 2 = 8 points (16 N/A: PROMPT_01..07, 11..18, 20)
  - Repetition / Termination: 20 prompts * 2 = 40 points (0 N/A)
  - **Reconciled Total**: Exactly **180 possible points** (Total points earned: 0 / 180 = 0.0%).
- **Subjective Scoring Notice**: Semantic judgments (e.g. relevance and task completion) are documented on evidence but formally require human review.

---

## 4. Repetition and Termination Analysis
- **Repetition Heuristics Recomputed**:
  - *Word Trigram Duplication (>=2 identical trigrams)*: 14 of 20 prompts (70.0%).
  - *Token Bigram Duplication (>=4 identical bigrams)*: 20 of 20 prompts (100.0%).
  - *Newline Runaway (>=10 consecutive newlines)*: 2 of 20 prompts (10.0%, PROMPT_17 and PROMPT_05).
  - *Combined Looping Rate*: 20 of 20 prompts (100.0%).
  - *Correction*: The prior claim that all 20 prompts exhibited phrase-level repetition was imprecise; at the word trigram level, 14/20 repeat, whereas at the subword/token level, 20/20 loop.
- **Termination Reason Semantics**:
  - In chakrview/runtime/neural_runtime_adapter.py, stop_reason initializes to MAX_TOKENS. If ctive_eos_stop is True and 
ext_token == 1 (EOS_ID), it switches to STOP_TOKEN and breaks early.
  - In all 20 runs, step iterated through all 48 steps without generating token 1 as the greedy argmax.
  - Reaching MAX_TOKENS proves that token 1 was not the top argmax token during greedy decoding, but does NOT establish whether the model can generate EOS under non-greedy sampling or with repetition penalties. Thus, EOS termination capability is correctly categorized as **INCONCLUSIVE**.

---

## 5. Evidence Classification: Fact vs. Hypothesis

1. **Directly Observed**:
   - Total points earned: 0/180.
   - Stop reason: MAX_TOKENS across all 20 outputs.
   - Token bigram repetition: 20/20 (100.0%).
   - Word trigram repetition: 14/20 (70.0%).
   - Math template bias (Step-by-Step Solution:): 6/20 (30.0%).
   - Checkpoint SHA-256 and tokenizer SHA-256 remain strictly intact.
   - In-memory weights remained immutable throughout inference (Delta-W = 0).
2. **Supported Hypothesis**:
   - The model defaults to math headers because the pretraining dataset contained a disproportionate volume of synthetic arithmetic problems.
   - The lack of conversational turn-taking stems from the absence of instruction-following SFT.
3. **Not Yet Tested**:
   - Performance under sampling decoding (	emperature > 0.0, 	op_k > 0, 	op_p < 1.0).
   - Repetition mitigation using presence/frequency penalties (
epetition_penalty > 1.0).
   - Instruction following with chat-specific role prefixes (<|user|>, <|assistant|>).
4. **Inconclusive**:
   - Complete absence of EOS emission under all decoding regimes.

---

## 6. Hash and Identity Clarifications
- The hash c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da is the parameter hash of a fresh-initialized ChakrMicro model initialized with 	orch.manual_seed(42). It is NOT a source code file hash.
- The source code file hash for chakrview/brain/model.py is 1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361.
