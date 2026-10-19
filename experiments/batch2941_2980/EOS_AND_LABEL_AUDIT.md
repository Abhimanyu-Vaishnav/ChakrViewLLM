# EOS and Label Audit: Batch 2941–2980
## Deep Investigation into Target Label Construction, Loss Masking & EOS Induction

### 1. Empirical First-Token Logit Audit

We compared the raw unnormalized logits emitted by the Canonical Step-10,000 baseline versus the Step-100 SFT candidate on benchmark case `INST_01`:
`"Write a single sentence explaining why water expands when it freezes."`

```text
Decision Position: Terminal Prompt Token (index 15, character '.')

Canonical Step-10,000 Baseline:
  Rank 1: Token 35 (' ')   -> logit 11.09
  Rank 2: Token 13 ('\n')  -> logit 10.24
  Rank 3: Token 41 ('&')   -> logit 9.68
  ...
  Rank 58: Token 1 (EOS)   -> logit -1.42  (Deeply suppressed)

Step-100 SFT Pilot Candidate:
  Rank 1: Token 1  (EOS)   -> logit 10.79  <-- GREEDY WINNER (Immediate Empty Output)
  Rank 2: Token 35 (' ')   -> logit 8.60
  Rank 3: Token 13 ('\n')  -> logit 8.59
```

### 2. Diagnosis: What Induced the EOS Over-Triggering?

1. **Target Label Construction in SFT Pilot**:
   In the previous pilot (Batch 2861–2900), every training sequence was constructed as:
   `target_seq = [-100]*len(prompt_tokens) + response_tokens + [EOS]`
   There was 0% representation of sequences that continue without immediately ending in `<EOS>`.
2. **Context Window Conditioning**:
   Because `ChakrMicro` has limited capacity (`2.5M` parameters), 100 optimizer steps on 100% terminated instruction sequences rapidly overwrote the terminal token transition probabilities. The model learned that after completing a grammatical clause ending with `.`, the probability mass collapses to `<EOS>`.
3. **Difference between `raw_completion` and `plain_instruction`**:
   Under `plain_instruction`, the prompt ends with `\nResponse:`. The model was trained specifically on that delimiter, so its logits correctly favor beginning response tokens (`'The'`, `'A'`).
   Under `raw_completion`, the prompt ends with standard punctuation (`.`, `?`), where the fine-tuned model immediately triggers `<EOS>`.

### 3. Label Mask Solution in Batch 2941–2980

To prevent this distortion without hacky runtime suppression:
- **Instruction Data**: Supervise `response_tokens + [EOS]`.
- **Replay Data**: Supervise `response_tokens` with `append_eos=False`.
  By training on open-ended text continuation pairs where the target sequence continues through natural words rather than ending in `<EOS>`, the model retains high probability mass for continuation tokens following periods and natural punctuation.
