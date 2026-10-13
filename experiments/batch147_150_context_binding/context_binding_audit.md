# Context Binding & Extractive Learning Audit: Batch 147–150

**Date:** 2026-10-13  
**Milestone:** Batch 147–150 Context Binding & Grounded Answers  
**Scope:** Causal Token Alignment, Key-Value Span Attention, and Delimiter Supervision  

---

## 1. Executive Summary

In Batch 143–146, we confirmed that supervised fine-tuning (SFT) successfully instills ChatML role formatting and eliminates wikitext artifacts, reducing instruction loss by >2.3 nats. However, qualitative generations showed that the 3.44M parameter neural core still struggles to bind to specific values in a supplied context, reverting to repetitive pretraining attractors when the context exceeds ~30 tokens.

This audit investigates the mechanistic and causal requirements for **Context Binding** (associating a requested key with its corresponding context value and reproducing it).

---

## 2. Ranked Diagnostic Findings

| Rank | Severity | Area | Finding / Mechanism | Impact & Remedy |
|:---|:---|:---|:---|:---|
| **1** | **High** | **Attention Span Horizon & Key-Value Distance** | In raw multi-turn SFT, the context tokens and the question prompt are separated by 30–80 tokens. In a small 6-layer transformer (`d_model=192`, 6 heads, head dim 32), self-attention entropy diffuses over long spans unless explicitly trained on tight key-value bindings. | **REMEDY:** Design a curriculum starting with concise key-value tuples (e.g. `Context: color is blue. Question: What is color? Answer: blue.`), then progressively introducing distractors and longer spans. |
| **2** | **Medium** | **Causal Alignment Verification** | `build_causal_sequence` alignment was verified: `input_seq[len(prompt)]` is the final prompt token (here `\n` after `<|im_start|>assistant\n`), predicting `response_seq[0]`. | **VERIFIED:** Causal next-token prediction is exactly aligned with zero off-by-one shifts. |
| **3** | **Medium** | **Supervision Density & Answer Token Weighting** | In short-answer QA (e.g. `"blue"` is 1 token, `<|im_end|>\n` is 9 tokens), the loss gradient is heavily weighted towards the delimiter. Without balancing, the model learns when to terminate, but under-learns the value token itself. | **REMEDY:** Ensure diverse value lengths and evaluate value token prediction accuracy independently of delimiter termination. |
| **4** | **Verified Safe** | **Frozen Tokenizer Parity** | Byte-Level BPE (`vocab_4096`) represents keys, values, and delimiters deterministically without unknown token insertion. | **VERIFIED.** |

---

## 3. Context Binding Specification

To evaluate real context binding (and prevent trivial memorization):
1. **Unseen Key-Value Combinations:** Held-out tests must evaluate keys and values whose combination never appeared in training.
2. **Distractor Invariance:** Prompts must include 1 to 2 irrelevant key-value pairs (e.g. `Context: city is Pune. animal is tiger. Question: What is city?`) to test selective attention.
3. **Negative / Unsupported Queries:** Prompts where the queried key is absent from the context must elicit explicit abstention (e.g. `"Not mentioned in context."`).
4. **Separation of Neural vs Deterministic Extraction:** A deterministic regex/key-value parser is provided as an engineering reference baseline, completely separated from the neural model's raw generated answer.
