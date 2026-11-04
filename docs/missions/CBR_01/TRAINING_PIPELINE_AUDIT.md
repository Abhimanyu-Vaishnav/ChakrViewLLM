# CBR-01 — WORKSTREAM A: TRAINING PIPELINE DEEP AUDIT

**Audit Scope:** Data encoding, causal shifting, loss masking, attention masking, optimizer dynamics, and generation decoding.  

---

## 1. End-to-End Data Flow

```
Raw Prompt & Response Strings
             ↓
[Prompt Contract Normalization] (Delimiter preservation & formatting)
             ↓
[Byte-Pair Tokenizer (4,096 vocab)] -> (p_toks, r_toks)
             ↓
[Causal Alignment & Shifting]
  Input:  [BOS] + p_toks + r_toks
  Target: [-100]*len(p_toks) + r_toks + [EOS]
             ↓
[Dynamic Batch Padding] -> Padded with pad_id=2 (Inputs) and -100 (Targets)
             ↓
[ChakrMicro Transformer Forward Pass] -> Logits [B, T, 4096]
             ↓
[CrossEntropyLoss with ignore_index=-100] -> Gradients computed solely on active response tokens
             ↓
[Gradient Clipping (max_norm=1.0) & Cosine LR Decay] -> AdamW Optimizer Step
```

---

## 2. Granular Investigation Findings

### 1. Off-by-One Alignment Check: PASS (Verified Mathematically & Experimentally)
In `chakrview/training/prompt_contract.py`:
- `input_seq = [BOS] + p_toks + r_toks` (Length $N$)
- `target_seq = [-100]*len(p_toks) + r_toks + [EOS]` (Length $N$)
Because autoregressive models shift logits internally by 1 ($input_t$ predicts $target_t$):
- $input_{len(p)}$ is the final prompt token. Its target in $target_{len(p)}$ is $r_0$ (the first token of the response).
- $input_{N-1}$ is the final response token $r_{-1}$. Its target in $target_{N-1}$ is $[EOS]$.
- Every prompt token is masked out with $-100$.
- **Conclusion:** Perfect mathematical causal alignment. Zero off-by-one errors.

### 2. Prompt Delimiter Discrepancy: DEFECT IDENTIFIED (MEDIUM SEVERITY)
- **Defect:** `SFTConfig.strip_prompt_delimiters` defaults to `True` in `sft_engine.py`, which strips whitespace from prompt strings (`prompt.strip()`).
- **Impact:** In BPE tokenization, `'Assistant:'` and `'Assistant:\n'` tokenize to completely different IDs. More critically, stripping trailing spaces causes the first response token to lose its leading space delimiter (e.g. `'Hello'` vs `' Hello'`), shifting token IDs between training time and runtime generation (`session.py` preserves prompts as written).
- **Correction:** Enforce `strip_prompt_delimiters=False` universally so training and autoregressive inference share bit-exact token prefixes.

### 3. Dynamic Padding & Attention Masking: PASS
- Inputs pad with `pad_id=2`, targets pad with `-100`.
- Loss correctly ignores $-100$.
- Causal triangular mask prevents any future information leakage across tokens.

### 4. Optimizer & Loss Normalization: PASS
- `CrossEntropyLoss(ignore_index=-100)` correctly averages loss solely over active response tokens ($N_{active} > 0$).
- AdamW configured with `weight_decay=0.01` and cosine decay with floor $5\times 10^{-6}$.
