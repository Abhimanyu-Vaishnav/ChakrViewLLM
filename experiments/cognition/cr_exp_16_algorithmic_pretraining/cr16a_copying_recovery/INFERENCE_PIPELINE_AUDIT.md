# Inference Pipeline Audit

## 1. Inference Mechanics
The benchmark evaluation assesses sequence copying via free autoregressive generation:
1. The model is primed with the exact prompt string: `p['prompt']`.
2. Greedy decoding selects the highest probability token:
   $$\hat{x}_{t+1} = \text{argmax}_k \text{logits}_t[k]$$
3. Generation halts when:
   - The token ID matches the EOS token (`1`), OR
   - The decoded token contains a newline character (`\n`), OR
   - The maximum new token limit (20 tokens) is reached.
4. The output string is sliced at the first newline:
   `output.strip().split('\n')[0].strip()`

## 2. Evaluation Discrepancies and Failure Modes
1. **Empty Output Bug:**
   - In Arm B, when evaluating on unaligned templates (e.g., `Echo sequence:` when Arm B was trained on `Copy: {seq} -> `), the model immediately generated `\n` or EOS, producing an empty string `""` and scoring 0.
2. **Infinite Subword Loops:**
   - In the Stage A pilot, the model generated repetitive unigrams (`north stone stone stone...`), repeating up to the max token limit.
3. **Decoded Byte Stream Safety:**
   - Decoding raw BPE tokens with `errors="replace"` prevents crashing on incomplete UTF-8 byte sequences during intermediate autoregressive steps.
4. **Greedy vs Temperature:**
   - Evaluator calibration verified that greedy decoding ($T=0$) is deterministic, reproducible, and provides the purest measurement of algorithmic capability.

