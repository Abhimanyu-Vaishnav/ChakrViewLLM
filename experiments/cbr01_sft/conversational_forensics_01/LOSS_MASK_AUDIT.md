# Loss Masking and Token Target Alignment Audit

## 1. Mathematical and Implementation Formulation

In `chakrview/training/prompt_contract.py`:
- `input_seq = [BOS] + prompt_tokens + response_tokens`
- `target_seq = [-100] * len(prompt_tokens) + response_tokens + [EOS]`

In `chakrview/training/sft_engine.py`:
- `logits = self.model(x)` (shape: `[B, T, vocab_size]`)
- `loss = self.loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))`

### Alignment Verification:
At position `t = len(prompt_tokens)`:
- `input_seq[t]` is the final token of the prompt (`prompt_tokens[-1]`).
- The causal transformer computes representation `h[t]` using `input_seq[0...t]` (i.e. BOS and the entire prompt).
- `logits[t]` predicts the next token.
- `target_seq[t]` is `response_tokens[0]` (the very first response token).
- For `t < len(prompt_tokens)`, `target_seq[t] == -100`, masking all prompt tokens from contributing to the loss.
- At `t = len(input_seq) - 1` (the last response token), `target_seq[t] == EOS` (1), teaching the model when to stop.

Conclusion on alignment: **Mathematically bit-exact and verified correct causal alignment.** No shift bug exists.

## 2. Empirical Verification on Training Examples

### Example Record 0 (Category: `explanations_reasoning`)
- Prompt Tokens: 20, Response Tokens: 40, Total Seq: 61
- Masked Target Tokens (-100): 20 (Exact match to prompt length: True)
- Active Target Tokens: 41 (Exact match to response + EOS: True)

| Pos | Input ID | Input Token | Target ID | Target Token | Masked? |
|---|---|---|---|---|---|
| 0 | 0 | '' | -100 | 'MASK(-100)' | True |
| 1 | 88 | 'U' | -100 | 'MASK(-100)' | True |
| 2 | 835 | 'ser' | -100 | 'MASK(-100)' | True |
| 3 | 324 | ': ' | -100 | 'MASK(-100)' | True |
| 4 | 90 | 'W' | -100 | 'MASK(-100)' | True |
| 5 | 107 | 'h' | -100 | 'MASK(-100)' | True |
| 6 | 725 | 'at ' | -100 | 'MASK(-100)' | True |
| 7 | 589 | 'is ' | -100 | 'MASK(-100)' | True |
| 8 | 707 | 'mo' | -100 | 'MASK(-100)' | True |
| 9 | 2325 | 'dul' | -100 | 'MASK(-100)' | True |
| 18 | 1855 | 'sist' | -100 | 'MASK(-100)' | True |
| 19 | 1807 | 'ant' | -100 | 'MASK(-100)' | True |
| 20 | 61 | ':' | 1655 | 'Mo' | False |
| 21 | 1655 | 'Mo' | 2325 | 'dul' | False |
| 22 | 2325 | 'dul' | 448 | 'ar ' | False |
| 23 | 448 | 'ar ' | 2072 | 'design ' | False |

### Example Record 10 (Category: `identity_awareness`)
- Prompt Tokens: 24, Response Tokens: 29, Total Seq: 54
- Masked Target Tokens (-100): 24 (Exact match to prompt length: True)
- Active Target Tokens: 30 (Exact match to response + EOS: True)

| Pos | Input ID | Input Token | Target ID | Target Token | Masked? |
|---|---|---|---|---|---|
| 0 | 0 | '' | -100 | 'MASK(-100)' | True |
| 1 | 88 | 'U' | -100 | 'MASK(-100)' | True |
| 2 | 835 | 'ser' | -100 | 'MASK(-100)' | True |
| 3 | 324 | ': ' | -100 | 'MASK(-100)' | True |
| 4 | 896 | 'In' | -100 | 'MASK(-100)' | True |
| 5 | 1148 | 'tro' | -100 | 'MASK(-100)' | True |
| 6 | 666 | 'duc' | -100 | 'MASK(-100)' | True |
| 7 | 267 | 'e ' | -100 | 'MASK(-100)' | True |
| 8 | 124 | 'y' | -100 | 'MASK(-100)' | True |
| 9 | 1339 | 'our' | -100 | 'MASK(-100)' | True |
| 22 | 1855 | 'sist' | -100 | 'MASK(-100)' | True |
| 23 | 1807 | 'ant' | -100 | 'MASK(-100)' | True |
| 24 | 61 | ':' | 992 | 'I ' | False |
| 25 | 992 | 'I ' | 1001 | 'am ' | False |
| 26 | 1001 | 'am ' | 2035 | 'ChakrView' | False |
| 27 | 2035 | 'ChakrView' | 276 | ', ' | False |

### Example Record 50 (Category: `context_grounding`)
- Prompt Tokens: 38, Response Tokens: 10, Total Seq: 49
- Masked Target Tokens (-100): 38 (Exact match to prompt length: True)
- Active Target Tokens: 11 (Exact match to response + EOS: True)

| Pos | Input ID | Input Token | Target ID | Target Token | Masked? |
|---|---|---|---|---|---|
| 0 | 0 | '' | -100 | 'MASK(-100)' | True |
| 1 | 88 | 'U' | -100 | 'MASK(-100)' | True |
| 2 | 835 | 'ser' | -100 | 'MASK(-100)' | True |
| 3 | 324 | ': ' | -100 | 'MASK(-100)' | True |
| 4 | 81 | 'N' | -100 | 'MASK(-100)' | True |
| 5 | 114 | 'o' | -100 | 'MASK(-100)' | True |
| 6 | 794 | 'tic' | -100 | 'MASK(-100)' | True |
| 7 | 1330 | 'e: ' | -100 | 'MASK(-100)' | True |
| 8 | 898 | 'The ' | -100 | 'MASK(-100)' | True |
| 9 | 2916 | 'caching ' | -100 | 'MASK(-100)' | True |
| 36 | 1855 | 'sist' | -100 | 'MASK(-100)' | True |
| 37 | 1807 | 'ant' | -100 | 'MASK(-100)' | True |
| 38 | 61 | ':' | 898 | 'The ' | False |
| 39 | 898 | 'The ' | 2916 | 'caching ' | False |
| 40 | 2916 | 'caching ' | 2365 | 'lay' | False |
| 41 | 2365 | 'lay' | 392 | 'er ' | False |

