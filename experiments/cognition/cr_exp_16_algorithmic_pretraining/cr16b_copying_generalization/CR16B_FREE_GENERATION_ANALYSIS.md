# CR-EXP-16B Free Generation Analysis

## 1. Free Generation Protocol
In free generation:
- The model is primed solely with the prompt string.
- Tokens are sampled greedily ($T=0$) until EOS (`1`), newline (`\n`), or max new tokens ($25$) is reached.
- The output is sliced strictly at the first newline character.

## 2. Qualitative Prediction Samples from Sealed Benchmark

### A. Successful In-Context Copying (Arm D)
- Prompt: `Echo sequence: kappa tau theta gamma nu\nOutput: `
  $\implies$ Prediction: `'kappa tau theta gamma nu'` (EXACT MATCH)
- Prompt: `Echo sequence: delta beta omicron zeta\nOutput: `
  $\implies$ Prediction: `'delta beta omicron zeta'` (EXACT MATCH)
- Prompt: `Repeat: dog\nOutput: `
  $\implies$ Prediction: `'dog'` (EXACT MATCH)
- Prompt: `Copy: tree -> `
  $\implies$ Prediction: `'tree'` (EXACT MATCH)
- Prompt: `Repeat the name: John\nName: `
  $\implies$ Prediction: `'John'` (EXACT MATCH)
- Prompt: `Copy: Victoria -> `
  $\implies$ Prediction: `'Victoria'` (EXACT MATCH)

### B. Typical Free-Generation Failure Modes
1. **Length Degradation / Truncation (CAT_E, CAT_F):**
   - Prompt: `Echo sequence: gamma xi rho omicron theta epsilon\nOutput: `
     $\implies$ Prediction: `'gamma xi rho omicron theta'` (Omitted the final token `'epsilon'`).
2. **Subword Hallucination (CAT_G, CAT_H):**
   - Prompt: `Repeat the name: Gautam\nName: `
     $\implies$ Prediction: `'Grace'` (First token `'G'` correct, second token drifted).
   - Prompt: `Repeat the name: Vikas\nName: `
     $\implies$ Prediction: `'Vihaan'` (First token `'Vi'` correct, second token drifted).
3. **Empty Output Bug (CAT_K on Unseen Templates):**
   - Prompt: `Replicate the following: apple bear\nExact copy: `
     $\implies$ Prediction: `''` (Model immediately emitted `\n` or EOS due to lack of unseen template exposure in single-template models).

