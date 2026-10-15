# Evaluation & Loss Masking Audit Report
**Location:** `experiments/batch381_410_integrity/reports/EVALUATION_AUDIT.md`  
**Milestone:** Batches 381–410  
**Scope:** Root-cause analysis of 0% exact-match and pseudo-termination in neural answer generation.

---

## 1. Executive Summary: The Two Root Defects

Auditing the generation and evaluation contract revealed two structural flaws that prevented ChakrMicro from producing real answers in Batches 311–380:

### Defect A: Pseudo `<|im_end|>` vs Real Token 1 (`<|endoftext|>`)
1. In the vocabulary (`release/chakrview-v0.5.0/data/experiments/vocab_4096`), `<|im_end|>` is **NOT** a reserved single special token.
2. It decomposes into **8 separate ASCII/BPE tokens**: `[63, 127, 1012, 98, 275, 103, 127, 65]` (`<`, `|`, `im`, `_`, `en`, `d`, `|`, `>`).
3. Model configuration explicitly defines `eos_token_id = 1` and `bos_token_id = 0`.
4. Prior training appended the string `"<|im_end|>"` to target responses, forcing the model to learn 8 arbitrary ASCII characters as an end delimiter instead of emitting true EOS token ID `1`.
5. During evaluation, greedy generation looked for the literal substring `"<|im_end|>"`. The model produced fragments like `'<imen|_d>'`, which evaluation scripts counted as "clean termination" (81%–100%) even though the model was emitting corrupted character noise.

### Defect B: Off-By-One Shift Alignment in `encode_chatml_example`
In `run_controlled_training.py` (lines 80–82):
```python
p_tokens = [0] + tokenizer.encode(prompt)
r_tokens = tokenizer.encode(response)
input_ids = p_tokens + r_tokens
labels = [-100] * (len(p_tokens) - 1) + r_tokens + [1]
```
1. Notice `input_ids` has length `len(p_tokens) + len(r_tokens)`.
2. But `labels` has length `(len(p_tokens) - 1) + len(r_tokens) + 1` = `len(p_tokens) + len(r_tokens)`.
3. However, `labels` inserted `r_tokens` starting at index `len(p_tokens) - 1`.
4. In next-token prediction, `shift_logits = logits[:, :-1]` is paired with `shift_labels = labels[:, 1:]`.
5. Because of the `- 1` in `[-100] * (len(p_tokens) - 1)`, the prompt's final token was paired to predict `r_tokens[0]`, but the last token in `labels` became `1` (EOS) without token `1` being in `input_ids`!
6. This caused the model to be penalized on the prompt's boundary and misaligned on the end-of-sequence token.

---

## 2. Corrected Target Masking & Sequence Contract

```python
# Mathematically exact next-token supervision:
p_tokens = [0] + tokenizer.encode(prompt)
r_tokens = tokenizer.encode(response) + [1]  # Token 1 is true EOS

input_ids = p_tokens + r_tokens
labels = [-100] * len(p_tokens) + r_tokens

assert len(input_ids) == len(labels)
shift_x = input_ids[:-1]
shift_y = labels[1:]

# In shift_x and shift_y:
# - Prompt tokens shift_x[0 : len(p_tokens)-1] are paired with labels[1 : len(p_tokens)], which are all -100 (ignored).
# - Prompt's final token shift_x[len(p_tokens)-1] directly predicts labels[len(p_tokens)] == r_tokens[0]!
# - The final response token predicts token 1 (EOS)!
```

---

## 3. Impact on Capability Claims

- The 81% "clean termination" reported in Batch 351–380 was superficial: it measured tokenization artifacts (`<imen|_d>`), not true semantic stopping.
- The 0% exact match was expected: the model was receiving corrupted labels at sequence boundaries and hallucinating BPE punctuation.
- Fixing this contract is the prerequisite to enabling true neural answer generation.
