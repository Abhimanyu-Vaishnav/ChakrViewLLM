# Batch 63 — Full Audit & Trace: Training-to-Generation Path

## 1. Executive Summary & Core Discovery

The investigation into the reported discrepancy—where teacher-forced cross-entropy loss dropped to ~0.38 while autoregressive greedy generation collapsed to `|277` (0/32 exact match)—identified **two concrete root causes**:

1. **Prompt/Response Delimiter Divergence (`.strip()` dropping `\n`)**:
   - In `SFTDataset.encode_example` (`chakrview/training/sft_engine.py`) and `DeterministicEvaluator.encode_record` (`experiments/batch42_reproducible_evaluation/evaluation/evaluator.py`):
     ```python
     p_toks = self.tokenizer.encode(prompt.strip())
     r_toks = self.tokenizer.encode(response.strip())
     ```
   - In ChatML, prompts terminate with `<|im_start|>assistant\n`.
   - Applying `.strip()` silently removes the trailing `\n`.
   - Tokenizer encoding of `prompt`:
     - Raw prompt: ends in token sequence `[..., 427 ('as'), 1855 ('sist'), 1807 ('ant'), 13 ('\n')]` (39 tokens + BOS = 40 tokens).
     - Stripped prompt: ends in `[..., 427 ('as'), 1855 ('sist'), 1807 ('ant')]` (38 tokens + BOS = 39 tokens). Token `13` (`\n`) is discarded!
   - As a result, in `input_seq = [BOS] + p_toks + r_toks`, the token immediately preceding the first response token is token `1807` (`ant`), NOT `13` (`\n`).
   - During training, the model was conditioned on prompt ending in `ant` to predict the response.
   - However, during inference evaluation (`evaluate_trained.py`), generation prompts were tokenized from the raw string ending in `\n`:
     ```python
     prompt_tokens = [0] + tokenizer.encode(r["prompt"]) # ends in token 13 ('\n')
     ```
   - When the trained model is given a prompt ending in `\n`, it encounters a state it was **never trained on**. At token `\n`, the model reverts to its pretraining Wikipedia table attractor, predicting token `127` (`|`), producing `|277`.
   - **Verification**: When evaluated using the exact prompt representation it was trained on (`prompt.strip()`), the model never emits `|277`; it immediately emits `27<|im_end|>`.

2. **Template Dominance Masking Task-Token Underfitting**:
   - In each synthetic training example (`response: "23<|im_end|>\n"`), the target sequence consists of:
     - 1 answer token (`23` or `2` + `7`), followed by
     - 8-9 template tokens (`<`, `|`, `im`, `_`, `en`, `d`, `|`, `>`, `\n`, `<EOS>`).
   - In 50 steps with batch size 2, the model quickly achieved near-zero loss (0.017) on the 8 template tokens common to every example.
   - However, the loss on the answer tokens remained high (~2.83).
   - Because 88% of target tokens were template tokens, the unweighted token loss averaged out to 0.38, creating the illusion of convergence while the model had only learned the modal prefix digit `2` and the ChatML closing tag.

---

## 2. Hand-Verifiable End-to-End Sequence Trace

We trace the canonical synthetic example `SYNTH-ADD-13-10`:
- **Raw JSON Record**:
  ```json
  {
    "id": "SYNTH-ADD-13-10",
    "prompt": "<|im_start|>user\nCompute: 13 + 10 = ?<|im_end|>\n<|im_start|>assistant\n",
    "response": "23<|im_end|>\n",
    "expected_answer": "23"
  }
  ```

### Step 1: Raw vs Stripped Tokenization

| String Component | String Fragment | Token ID | Decoded Token | Notes |
|:---|:---|:---|:---|:---|
| BOS | `""` | 0 | `""` | Prepended to input_seq |
| Prompt | `<` | 63 | `'<'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `im` | 1012 | `'im'` | |
| Prompt | `_` | 98 | `'_'` | |
| Prompt | `star` | 1832 | `'star'` | |
| Prompt | `t` | 119 | `'t'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `>` | 65 | `'>'` | |
| Prompt | `user` | 1940 | `'user'` | |
| Prompt | `\n` | 13 | `'\n'` | |
| Prompt | `Comp` | 1574 | `'Comp'` | |
| Prompt | `ute` | 1356 | `'ute'` | |
| Prompt | `: ` | 324 | `': '` | |
| Prompt | `13` | 1105 | `'13'` | |
| Prompt | ` + ` | 543 | `' + '` | |
| Prompt | `10` | 545 | `'10'` | |
| Prompt | ` = ` | 400 | `' = '` | |
| Prompt | `?` | 66 | `'?'` | |
| Prompt | `<` | 63 | `'<'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `im` | 1012 | `'im'` | |
| Prompt | `_` | 98 | `'_'` | |
| Prompt | `en` | 275 | `'en'` | |
| Prompt | `d` | 103 | `'d'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `>` | 65 | `'>'` | |
| Prompt | `\n` | 13 | `'\n'` | |
| Prompt | `<` | 63 | `'<'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `im` | 1012 | `'im'` | |
| Prompt | `_` | 98 | `'_'` | |
| Prompt | `star` | 1832 | `'star'` | |
| Prompt | `t` | 119 | `'t'` | |
| Prompt | `|` | 127 | `'|'` | |
| Prompt | `>` | 65 | `'>'` | |
| Prompt | `as` | 427 | `'as'` | |
| Prompt | `sist` | 1855 | `'sist'` | |
| Prompt | `ant` | 1807 | `'ant'` | **Last token when prompt.strip() is used (len=38)** |
| Prompt (raw) | `\n` | 13 | `'\n'` | **Present in raw prompt (len=39), DROPPED by .strip()** |
| Response | `23` | 984 | `'23'` | Target answer |
| Response | `<` | 63 | `'<'` | Template closing |
| Response | `|` | 127 | `'|'` | Template closing |
| Response | `im` | 1012 | `'im'` | Template closing |
| Response | `_` | 98 | `'_'` | Template closing |
| Response | `en` | 275 | `'en'` | Template closing |
| Response | `d` | 103 | `'d'` | Template closing |
| Response | `|` | 127 | `'|'` | Template closing |
| Response | `>` | 65 | `'>'` | Template closing |
| Response (raw) | `\n` | 13 | `'\n'` | Present in raw response, DROPPED by .strip() |
| EOS | `""` | 1 | `""` | Appended to target_seq |

---

### Step 2: Tensor Alignment in Batch 59-62 SFT Engine

In the Batch 59-62 implementation:
- `p_toks = tokenizer.encode(prompt.strip())` -> 38 tokens
- `r_toks = tokenizer.encode(response.strip())` -> 9 tokens
- `input_seq = [0] + p_toks + r_toks` -> length 48 tokens (0..47)
- `target_seq = ([-100] * len(p_toks)) + r_toks + [1]` -> length 48 tokens (0..47)

Position-by-position alignment around boundary:
```
Position 37: input = 1855 ('sist'), target = -100 (MASK)
Position 38: input = 1807 ('ant' ), target =  984 ('23')  <-- Logit at pos 38 predicts target 984!
Position 39: input =  984 ('23'  ), target =   63 ('<' )  <-- Logit at pos 39 predicts target 63
Position 40: input =   63 ('<'   ), target =  127 ('|' )
Position 41: input =  127 ('|'   ), target = 1012 ('im')
Position 42: input = 1012 ('im'  ), target =   98 ('_' )
Position 43: input =   98 ('_'   ), target =  275 ('en')
Position 44: input =  275 ('en'  ), target =  103 ('d' )
Position 45: input =  103 ('d'   ), target =  127 ('|' )
Position 46: input =  127 ('|'   ), target =   65 ('>' )
Position 47: input =   65 ('>'   ), target =    1 (EOS )  <-- Logit at pos 47 predicts EOS
```

Notice:
1. `pos 38` input is `ant`. Target is `23`. The model learns: $P(\text{tok} \mid \text{BOS} \dots \text{assistant})$.
2. During inference, `evaluate_trained.py` tokenized `prompt` without `.strip()`, creating an input of length 40:
   - Position 38: input = `ant`
   - Position 39: input = `\n` (token 13)
3. Greedy decoding queries the logit at `logits[0, -1, :]` (position 39, token `\n`).
4. At token `\n`, the model's highest logit is token `127` (`|`), which is its baseline pretraining attractor!

---

### Step 3: Empirical Validation of the Divergence

1. **Greedy decoding from raw prompt (ending in `\n`)**:
   - `input_tokens = [0] + tokenizer.encode(prompt)` (ends in `\n`)
   - Next token prediction: token 127 (`|`), followed by 53 (`2`), 58 (`7`), 58 (`7`), 63 (`<`), 127 (`|`), 1012 (`im`)...
   - Result: `|277<|im_end|>` (collapse to pretraining Wikipedia table header artifact).
2. **Greedy decoding from stripped prompt (ending in `ant`)**:
   - `input_tokens = [0] + tokenizer.encode(prompt.strip())` (ends in `ant`)
   - Next token prediction: token 53 (`2`), followed by 58 (`7`), 63 (`<`), 127 (`|`), 1012 (`im`), 98 (`_`), 275 (`en`), 103 (`d`), 127 (`|`), 65 (`>`), 1 (EOS).
   - Result: `27<|im_end|>` (NO `|` token!).

---

### Step 4: Tokenizer Granularity Analysis for Digits

Inspection of the vocabulary artifacts (`vocab_4096`) reveals that two-digit integers do not have uniform single-token representations:
- `20` -> single token `567` (`'20'`)
- `22` -> single token `1295` (`'22'`)
- `23` -> single token `984` (`'23'`)
- `26` -> single token `568` (`'26'`)
- `28` -> single token `3661` (`'28'`)
- BUT:
- `21` -> two tokens: `[53, 52]` (`'2'`, `'1'`)
- `27` -> two tokens: `[53, 58]` (`'2'`, `'7'`)
- `31` -> two tokens: `[54, 52]` (`'3'`, `'1'`)

When the model is trained with weak capacity (50 steps, lr=1e-4), predicting token `53` (`'2'`) gives a non-zero probability for all answers in the 20s. Once token `53` (`'2'`) is emitted, the subsequent token is conditioned on `'2'`. In the training dataset, `27` appeared with frequency, and the model defaulted to completing `'2'` with `'7'`.

---

## 3. The Correct Alignment Architecture

To fix the training-generation gap permanently and cleanly:
1. **Never strip semantic ChatML delimiters**: ChatML prompts must consistently end with `<|im_start|>assistant\n`.
2. **Preserve full prompt and response tokenization**:
   - Prompt tokens: `tokenizer.encode(prompt)` (preserving `\n`).
   - Input sequence: `[BOS] + p_toks + r_toks`.
   - Target sequence: `[-100] * len(p_toks) + r_toks + [EOS]`.
   - Last prompt token is `\n` (token 13).
   - Logit at position of `\n` predicts `r_toks[0]`.
3. **Inference parity**:
   - Generation input: `[BOS] + tokenizer.encode(prompt)`.
   - Greedy decoding predicts from the logit of `\n`, directly outputting the first response token.
