# SFT and ChatML Infrastructure Audit: Batch 143–146

**Date:** 2026-10-13  
**Milestone:** Batch 143–146 Instruction Learning & Grounded Comprehension  
**Scope:** `chakrview/runtime/conversation.py`, `chakrview/training/prompt_contract.py`, `chakrview/training/sft_engine.py`  

---

## 1. Executive Summary

Before introducing new instruction-tuning data or training runs, this audit inspects the mathematical, token-level, and architectural mechanisms of ChakrView's existing SFT and ChatML infrastructure. 

We examined:
1. Delimiter representations in the frozen 4,096-token vocabulary.
2. Causal shift and prompt loss-masking logic.
3. Supervision alignment between training and autoregressive generation.
4. Truncation and EOS handling.
5. Trust gating and dataset contamination checks.

---

## 2. Ranked Audit Findings

| Rank | Severity | Component | Finding | Evidence / Impact | Status |
|:---|:---|:---|:---|:---|:---|
| **1** | **High (Subtle Shift)** | `ConversationTemplate.encode_multiturn_sft` | **Active Token Shift in Target Alignment** | In `encode_multiturn_sft` (lines 160–165 of `conversation.py`), `target_tokens` was constructed such that the first assistant target token was shifted by one position relative to `input_tokens[1:]`. In contrast, `prompt_contract.build_causal_sequence` provides exact causal alignment where `input_tokens[len(prompt)]` predicts `response_tokens[0]`. | **FIXED / STANDARDIZED:** Standardized on `build_causal_sequence` semantics for all single-turn and multi-turn instruction formatting. |
| **2** | **Medium (Architecture Invariant)** | Frozen BPE Tokenizer (`vocab_4096`) | **Delimiters are Subwords, Not Atomic Special Tokens** | The ChatML delimiters (`<|im_start|>` and `<|im_end|>`) are not single out-of-vocabulary token IDs. `<|im_start|>` decomposes into 7 subwords: `['<', '|', 'im', '_', 'star', 't', '|', '>']`. | **CONFIRMED ARCHITECTURAL INVARIANT:** The frozen tokenizer cannot be modified. The subword decomposition is consistent between training and inference as long as strings are preserved without accidental tokenization boundary merges. |
| **3** | **Medium (Supervision Leakage)** | `sft_engine.py` / Prompt Delimiters | **Prompt-Format Dominance Risk** | If short responses (e.g. 5 tokens) are supervised alongside long delimiters (`<|im_end|>\n` is 8 tokens + EOS), delimiter tokens represent >60% of the active loss gradient. This risks inducing premature termination (EOS collapse) or repetitive delimiter loops. | **REMEDIATED:** Balanced response targets and evaluated terminal EOS loss separately from content token loss. |
| **4** | **Verified Safe** | `sft_engine.py` | **Trust Gate & Contamination Prevention** | `SFTDataset` strictly enforces `review_status != PENDING_HUMAN_REVIEW` and blocks `HELD_OUT_BENCHMARK` records from entering training. | **VERIFIED:** Zero train-test contamination observed. |
| **5** | **Verified Safe** | `InferenceEngine` Greedy Decoding | **Generation Delimiter Parity** | Autoregressive generation from `<|im_start|>assistant\n` matches the training prompt suffix bit-for-bit. | **VERIFIED.** |

---

## 3. Detailed Token-Level Analysis: Subword Delimiter Tokenization

Under the canonical frozen BPE tokenizer (`vocab_size=4096`), encoding `<|im_start|>assistant\n` yields:
```
Token IDs: [63, 127, 1012, 98, 1832, 119, 127, 65, 427, 1855, 1807, 13]
Subwords:  ['<', '|', 'im', '_', 'star', 't', '|', '>', 'as', 'sist', 'ant', '\n']
```
Encoding `<|im_end|>\n` yields:
```
Token IDs: [63, 127, 1012, 98, 275, 103, 127, 65, 13]
Subwords:  ['<', '|', 'im', '_', 'en', 'd', '|', '>', '\n']
```

### Implications:
- The model must learn the subword composition of delimiters through causal attention.
- When generating responses, generation terminates when the model produces either the canonical EOS token (ID `1`) or the delimiter `<|im_end|>`.
- Because `<|im_end|>` spans 8 tokens, greedy decoding can occasionally get stuck repeating partial delimiter tokens (e.g., `<|im_im_im_`). Loss masking must ensure adequate supervision on both semantic content and the closing delimiter.

---

## 4. Remediation Standard for Batch 143–146

1. **Exact Alignment Standard:** Use `build_causal_sequence` with `strip_delimiters=False` to ensure every token prediction is causally aligned.
2. **Loss Masking Invariant:** Prompt tokens (system message, user turn, `<|im_start|>assistant\n`) are assigned `ignore_index = -100`. Only the assistant's answer tokens and terminating `<|im_end|>\n` (plus EOS) are backpropagated.
3. **Conservative Optimization:** Apply the verified Batch 139–142 recipe (linear warmup from 0 to conservative peak LR, cosine decay to $2\times 10^{-5}$) to prevent disruption of the base 10,000-step weights.
