# Raw-Completion Regression Analysis: Batch 2901–2940
## Root-Cause Investigation: Immediate EOS / Empty Generations under Unformatted Prompts

### 1. Observed Empirical Anomaly

During evaluation of the authorized 100-step SFT pilot candidate (`sft_checkpoint_step_100.pt`) on the Batch 2781 Generation Benchmark (28 cases):
- Under `plain_instruction` mode (`Instruction: {prompt}\nResponse:`): **0 / 28 cases were empty (0.0%)**. The model generated non-empty text across all cases.
- Under `raw_completion` mode (unformatted prompt tokens directly): **20 / 28 cases terminated immediately with 0 generated tokens (71.4%)**, compared to **0 / 28 empty outputs (0.0%)** on the Canonical Step-10,000 baseline.

Furthermore, across the intermediate SFT checkpoints:
- `canonical_step_10000`: 0 / 28 raw empty (0.0%)
- `sft_step_25`: 5 / 28 raw empty (17.9%)
- `sft_step_50`: 19 / 28 raw empty (67.9%)
- `sft_step_75`: 20 / 28 raw empty (71.4%)
- `sft_step_100`: 20 / 28 raw empty (71.4%)

The regression scaled progressively with fine-tuning step count.

---

### 2. Logit-Level Root-Cause Verification

We examined the unnormalized output logits at the final prompt token position across the baseline and candidate models for prompt `INST_01`:
`"Write a single sentence explaining why water expands when it freezes."`

```text
Prompt: RAW CONTINUATION (Unformatted)
  Canonical Step-10,000:
    Rank 1: Token 35 (' ')   -> logit 11.09
    Rank 2: Token 13 ('\n')  -> logit 10.24
    Rank 3: Token 41 ('&')   -> logit 9.68
    (Token 1 / EOS logit is suppressed, ranking > 50)

  SFT Candidate (Step 100):
    Rank 1: Token 1  (EOS)   -> logit 10.79  <-- GREEDY ARGMAX WINNER
    Rank 2: Token 35 (' ')   -> logit 8.60
    Rank 3: Token 13 ('\n')  -> logit 8.59

Prompt: PLAIN INSTRUCTION ("Instruction: ...\nResponse:")
  Canonical Step-10,000:
    Rank 1: Token 13 ('\n')  -> logit 10.96
    Rank 2: Token 41 ('&')   -> logit 6.72
    Rank 3: Token 61 (':')   -> logit 6.63

  SFT Candidate (Step 100):
    Rank 1: Token 898 ('The ') -> logit 8.20  <-- NORMAL CONTINUATION
    Rank 2: Token 68  ('A')    -> logit 6.90
    Rank 3: Token 74  ('G')    -> logit 6.70
```

### 3. Root Cause vs. Hypotheses

| Factor | Status | Evidence |
|---|---|---|
| **Stop-Token / Runtime Bug** | **Refuted** | The runtime decoder correctly checks `next_token == 1` and marks `stop_reason = "STOP_TOKEN"`. |
| **Token Truncation / Overlength** | **Refuted** | Prompt length was only ~15 tokens, well within context limit 512. |
| **Decoding Config Mismatch** | **Refuted** | Both baseline and candidate ran under identical greedy decoding (`temperature=0.0`). |
| **Template Conditioning Overfitting** | **PROVEN** | Every single one of the 192 training examples was strictly formatted with `Instruction: ...\nResponse: ... <EOS>`. The model never saw raw natural language continuations ending without `<EOS>` during fine-tuning. Because fine-tuning lacked unconditioned replay data, the model learned that terminal punctuation (`.`, `?`, `:`) outside of an active instruction block signals an immediate sequence boundary (`<EOS>`). |

### 4. Mitigation Strategy for Future SFT Runs

1. **Do NOT Suppress EOS at Inference**:
   Hardcoding an artificial rule to mask or ban token 1 during generation would mask the training artifact rather than solve the root cause.
2. **Incorporate Pretraining Replay / Continuation Pairs**:
   Future SFT data mixtures must include ~20–30% unformatted document continuation pairs where target tokens do not end immediately with `<EOS>`, maintaining the model's natural continuation priors while learning instruction adherence.
