# ChakrView Normal Chat Gate v0.1 -- Master Report

**Date**: 2026-10-22
**Evaluation Path**: experiments/normal_chat_gate_v01/
**Verdict**: BASELINE VALIDATED
*(The gate establishes a verified, internally consistent, non-training baseline with Delta-W = 0. Records and calculations are reconciled. Note: Baseline validation certifies measurement validity, not model capability).*

---

## 1. Benchmark Objective and Scope
The purpose of Normal Chat Gate v0.1 is to establish an unassisted, non-training baseline for the canonical ChakrMicro model (step_10000, 3,443,136 parameters) across 20 foundational conversational tasks spanning factual QA, instruction following, context retention, reference resolution, abstention, arithmetic, summarization, and repetition resistance.

---

## 2. Evaluation Rubric and Scoring Definitions

Each applicable dimension is evaluated on a strict 0-2 point scale:
- **0 Points**: Complete failure, degenerate repetition, hallucinated irrelevant output, or runaway generation.
- **1 Point**: Partial relevance or fragmented intent, but defective execution, severe hallucination, or runaway syntax.
- **2 Points**: Accurate, relevant, properly bounded response adhering strictly to instructions.

### Applicability Rules:
- **Relevance**: Evaluated for all 20 prompts (Max: 40 points).
- **Instruction Following**: Evaluated for all 20 prompts (Max: 40 points).
- **Grounding in Context**: Evaluated only for prompts with an explicit context passage (PROMPT_05, PROMPT_06, PROMPT_07, PROMPT_10, PROMPT_17, PROMPT_18). Excluded as N/A for 14 prompts without context (Max: 12 points).
- **Task Completion / Reasoning**: Evaluated for all 20 prompts (Max: 40 points).
- **Appropriate Uncertainty**: Evaluated only for prompts testing abstention, unanswerability, or ambiguity (PROMPT_08, PROMPT_09, PROMPT_10, PROMPT_19). Excluded as N/A for 16 clear factual/instructional prompts (Max: 8 points).
- **Repetition / Termination**: Evaluated for all 20 prompts (Max: 40 points).

**Total Programmatic Denominator**: Exactly **180 possible points** (resolving prior 196 vs 204 discrepancies).
*Subjective Judgment Notice*: Semantic dimensions represent structured evaluation and require human review for confirmation.

---

## 3. Results Summary

- **Prompts Attempted**: 20 / 20 (100.0%)
- **Prompts Completed**: 20 / 20 (100.0%)
- **Total Points Earned**: 0 / 180 possible points (0.0%)
- **Relevance**: 0 / 40 (0.0%)
- **Instruction Following**: 0 / 40 (0.0%)
- **Grounding in Context**: 0 / 12 (0.0%)
- **Task Completion / Reasoning**: 0 / 40 (0.0%)
- **Appropriate Uncertainty**: 0 / 8 (0.0%)
- **Repetition / Termination**: 0 / 40 (0.0%)
- **Degenerate Repetition Rate**:
  - Word Trigram Repetition (>=2 duplicate trigrams): 14 / 20 (70.0%)
  - Token Bigram Repetition (>=4 duplicate bigrams): 20 / 20 (100.0%)
  - Newline Runaway (>=10 consecutive newlines): 2 / 20 (10.0%)
  - Combined Looping / Runaway Rate: 20 / 20 (100.0%)
- **Termination Reason**: 20 / 20 terminated on MAX_TOKENS limit (48 tokens). 0 / 20 emitted EOS_ID (token ID 1).
- **Average Inference Latency**: ~980 ms per prompt (CPU)
- **Weight Immutability (Delta-W = 0)**: 100% verified across all prompts

---

## 4. Priority Bottlenecks and Targeted Improvement Recommendations

1. **Pretraining Domain Collapse and Absence of Conversation SFT**:
   - *Directly Observed*: The model defaults to memorized pretraining math headers (Step-by-Step Solution:\nThe total number of...) across factual, conversational, and instruction prompts.
   - *Recommendation*: Conduct a dedicated conversation SFT stage with explicit role delimiters and turn boundaries (exp17_instruction_sft_v1).
2. **Greedy Argmax Token Budget Saturation**:
   - *Directly Observed*: 20 out of 20 outputs hit MAX_TOKENS without generating EOS token 1 under greedy decoding (	emp=0.0).
   - *Inconclusive*: Whether the model can emit EOS under nonzero temperature or repetition penalties has not been evaluated.
   - *Recommendation*: Train explicit EOS turn-boundary termination and evaluate sampling decoding.
3. **Structured Context Conditioning and Pointer Guidance**:
   - *Directly Observed*: Context-supplied facts (e.g. vehicle model Delta-X in PROMPT_18) fail to condition output generation, producing irrelevant ungrounded tokens.
   - *Recommendation*: Ground chat generation with context-constrained pointer heads (as developed in Exp16 v3.1).
