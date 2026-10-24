# ChakrView Normal Chat Gate v0.1 -- Failure Analysis and Diagnosis

## 1. Evidence-Based Observations Across All 20 Prompts

In the 20-prompt chat capability gate, the unassisted ChakrMicro step_10000 model exhibited 0% task completion under greedy decoding (	emperature=0.0, max_new_tokens=48).

### Failure Modes Observed:
1. **Pervasive N-Gram Looping**:
   - Word-level trigram repetition (>=2 occurrences) occurred in 14/20 prompts (70.0%).
   - Token-level bigram repetition (>=4 occurrences) occurred in 20/20 prompts (100.0%).
   - Typical loops:
     - of the city of the city of the city... (Prompts 01, 03, 04, 06, 08, 13, 16, 19, 20)
     - of the total of the total of the total... (Prompts 04, 09, 15)
     - of the world of the world of the world... (Prompts 05, 10, 12, 14, 18)
2. **Synthetic Math Template Bias**:
   - 6 of 20 prompts (Prompts 01, 04, 09, 15, 16, 19) prefixed generation with the exact template:
     Step-by-Step Solution:\nThe total number of...
   - Even when prompted with arithmetic (What is 17 plus 25?), it produced a formulaic loop rather than computation:
     Step-by-Step Solution:\nThe total number of the total of the total of the total...
3. **Repeated Newline Runaway**:
   - Prompt 17 (Summarization) generated 48 consecutive newline characters (\n\n\n...). Prompt 05 and 07 also exhibited initial newline bursts.
4. **Token Generation Limit Reached (MAX_TOKENS)**:
   - 20 of 20 outputs (100.0%) reached the 48-token generation limit (stop_reason: MAX_TOKENS). Token ID 1 (EOS_ID) was never selected as the top argmax token.

---

## 2. Root Cause Classification: Evidence Taxonomy

| Potential Bottleneck | Status | Classification | Evidence and Reasoning |
| :--- | :--- | :--- | :--- |
| **Pretraining Arithmetic Bias** | **Directly Observed** | Critical Bottleneck | Output texts explicitly reproduce pretraining synthetic math headers (Step-by-Step Solution:) on conversational prompts. |
| **Token Budget Exhaustion (MAX_TOKENS)** | **Directly Observed** | Critical Behavior | All 20 prompts generated exactly 48 tokens and ceased due to the runtime step counter hitting ctive_max_new. |
| **Token-Level Bigram Looping** | **Directly Observed** | Critical Defect | 100% of outputs contain >=4 duplicate token bigrams under greedy decoding. |
| **Pretraining Synthetic Corpus Overfit** | **Supported Hypothesis** | Root Cause Hypothesis | Model step 10,000 behavior strongly correlates with high-frequency n-grams in the synthetic pretraining curriculum. |
| **Absence of SFT Instruction Alignment** | **Supported Hypothesis** | Root Cause Hypothesis | The model has only undergone self-supervised pretraining; prompt formats like Instruction: ... Response: have never been trained. |
| **Inability to Emit EOS under Any Sampling** | **INCONCLUSIVE** | Untested Hypothesis | Under greedy decoding (	emp=0.0), token 1 was never emitted. Whether nonzero temperature or repetition penalties would sample token 1 remains untested. |
| **Tokenizer Encoding Correctness** | **Directly Observed** | Validated (Not a Factor) | 0 encoding mismatches; all prompt tokens and special token IDs match specifications. |
| **Runtime Weight Immutability** | **Directly Observed** | Validated (Not a Factor) | In-memory weight hash remained strictly identical before and after each inference step (Delta-W = 0). |
