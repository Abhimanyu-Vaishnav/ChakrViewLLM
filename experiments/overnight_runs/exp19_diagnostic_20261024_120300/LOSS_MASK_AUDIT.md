# ChakrView Exp19 — SFT Loss-Mask & Token Boundary Verification Audit

**Audited Component**: `chakrview/training/sft_engine.py` & `chakrview/training/prompt_contract.py`  
**Timestamp**: 2026-10-24T12:04:00  
**Status**: `VERIFIED CORRECT & SECURE`  

---

## 1. Objective of the Loss-Mask Audit

To formally inspect the supervised fine-tuning loss-mask construction and verify:
1. Prompts tokens (including formatting tokens and delimiters) are completely excluded from the supervised loss (`loss_mask = -100`).
2. Target supervision strictly begins at response token 0 (the first token of the target completion).
3. The final response token correctly supervises the prediction of the `<|endoftext|>` token (EOS token ID `1`).
4. Padding tokens added during dynamic batching receive `target = -100` and do not contribute to gradient updates.
5. In causal autoregressive attention, step $t$ accurately predicts target $t$ without future leakage or off-by-one phase shifts.

---

## 2. Token-by-Token Sequence Construction

In [chakrview/training/prompt_contract.py](file:///d:/Project/ChakrView/chakrview/training/prompt_contract.py#L33):

```python
def build_causal_sequence(
    prompt_tokens: List[int],
    response_tokens: List[int],
    bos_id: int = 0,
    eos_id: int = 1,
    ignore_index: int = -100,
    append_eos: bool = True,
) -> Tuple[List[int], List[int]]:
    if append_eos:
        input_seq = [bos_id] + prompt_tokens + response_tokens
        target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens + [eos_id]
```

### Mathematical & Algorithmic Trace

Let:
- $P = [p_0, p_1, \dots, p_{M-1}]$ be the prompt tokens of length $M$.
- $R = [r_0, r_1, \dots, r_{N-1}]$ be the response tokens of length $N$.
- $\text{BOS} = 0$, $\text{EOS} = 1$, $\text{IGNORE} = -100$.

Then:
$$\text{input\_seq} = [0, p_0, p_1, \dots, p_{M-1}, r_0, r_1, \dots, r_{N-1}]$$
$$\text{target\_seq} = [\underbrace{-100, -100, \dots, -100}_{M \text{ times}}, r_0, r_1, \dots, r_{N-1}, 1]$$

Both sequences have identical length:
$$\text{len}(\text{input\_seq}) = 1 + M + N = \text{len}(\text{target\_seq})$$

### Alignment Verification across Positions $t \in [0, M + N]$

In PyTorch causal cross-entropy loss, the model computes logits $L_t = f(\text{input\_seq}_{\le t})$ and computes loss against target $Y_t = \text{target\_seq}_t$:

| Position $t$ | Input Token $\text{input\_seq}_t$ | Target Token $\text{target\_seq}_t$ | Supervised? | Semantic Role |
| :---: | :--- | :--- | :---: | :--- |
| $t = 0$ | $\text{BOS}$ (`0`) | $-100$ | **NO** (Masked) | Model attends to BOS; target is masked. |
| $t = 1$ | $p_0$ | $-100$ | **NO** (Masked) | Attends to [BOS, $p_0$]; target is masked. |
| $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| $t = M-1$ | $p_{M-2}$ | $-100$ | **NO** (Masked) | Attends to prompt prefix; target is masked. |
| **$t = M$** | **$p_{M-1}$** (final prompt token) | **$r_0$** (first response token) | **YES** | **Model sees full prompt up to `Response:`; predicts 1st response token $r_0$!** |
| $t = M + 1$ | $r_0$ | $r_1$ | **YES** | Attends to prompt + $r_0$; predicts 2nd response token $r_1$. |
| $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| $t = M + N - 1$ | $r_{N-2}$ | $r_{N-1}$ | **YES** | Predicts final response token. |
| **$t = M + N$** | **$r_{N-1}$** (final response token) | **$1$** (EOS token ID) | **YES** | **Model sees full response; predicts `<|endoftext|>`!** |

---

## 3. Dynamic Batching and Padding Audit

In [chakrview/training/sft_engine.py](file:///d:/Project/ChakrView/chakrview/training/sft_engine.py#L214-L229):

```python
max_len = max(len(x) for x in batch_inputs)
pad_id = 2
ignore_index = -100

padded_inputs = []
padded_targets = []
for inp, tgt in zip(batch_inputs, batch_targets):
    pad_amt = max_len - len(inp)
    padded_inputs.append(inp + [pad_id] * pad_amt)
    padded_targets.append(tgt + [ignore_index] * pad_amt)
```

1. **Pad Token ID**: `pad_id = 2` (`<|pad|>`) is used for input padding.
2. **Target Masking**: Targets are padded strictly with `ignore_index = -100`.
3. **Loss Function**: `nn.CrossEntropyLoss(ignore_index=-100)`.
4. **Active Token Accounting**:
   $$\text{active\_tokens} = \sum (Y \neq -100) = \sum_{b} (N_b + 1)$$
   The exact count equals the number of response tokens plus 1 (for EOS). No prompt or padding tokens enter the loss denominator or compute gradients.

---

## 4. Empirical Test Verification

A unit test was executed validating the exact boundary:
```
Input : [0, 10, 11, 12, 20, 21]
Target: [-100, -100, -100, 20, 21, 1]
step 0: sees 0 -> predicts -100
step 1: sees 10 -> predicts -100
step 2: sees 11 -> predicts -100
step 3: sees 12 (final prompt token) -> predicts 20 (first response token)
step 4: sees 20 -> predicts 21
step 5: sees 21 -> predicts 1 (EOS)
```

**Verdict**: The loss mask and token boundary implementation is 100% mathematically correct, causal, and prevents prompt token leakage into gradient updates.
