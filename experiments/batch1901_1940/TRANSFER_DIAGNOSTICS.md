# CHAKRVIEW — Batches 1901–1940
## Practical Transfer Diagnostics & Capability Verification Report

### Executive Summary

To determine whether language-modeling perplexity improvements correspond to genuine downstream capabilities, the canonical baseline and all continued-pretraining checkpoints were evaluated against a frozen suite of **13 practical transfer tasks** spanning arithmetic, multi-step calculation, Python code completion, architectural propositions, and conversational Hindi/Hinglish continuations.

The empirical findings are definitive:
- **Canonical Model (Step 10,000):** Achieved **0 / 13 (0.00%)** accuracy.
- **150-Step Balanced CPT Checkpoints (Seeds 42, 123, 999):** Each achieved **1 / 13 (7.69%)** accuracy.
- **300-Step Extended CPT Checkpoint (Seed 42):** Achieved **1 / 13 (7.69%)** accuracy.
- **Root Cause of Single Pass:** The sole passing task across all adapted checkpoints was `class Queue:\n    def __init__(self):\n        self.items = ` predicting `[` (token matching prefix `[]`).
- **Conclusion:** Continued pretraining produces **zero measurable improvement** in arithmetic calculation, multi-step reasoning, proposition recall, or syntax completion beyond single-bracket closure. Lower cross-entropy on language modeling does **NOT** translate into task reasoning or computational competence.

---

### 1. Diagnostic Suite Specification and Scoring Rules

- **Decoding Strategy:** Greedy argmax on the first generated token from the language model head (`torch.argmax(logits[0, -1, :])`).
- **Scoring Rule:** Exact match between decoded token string and expected completion, or non-empty prefix match of expected target.
- **Execution Rule:** Neural model completions only. Tool assistance via `SafeCalculator` or deterministic Python sandbox is strictly disallowed.

---

### 2. Task-by-Task Diagnostic Results

| Category | Task Prompt | Expected Target | Canonical Prediction | CPT 150-Step Prediction | CPT 300-Step Prediction | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arithmetic** | `15 + 25 = ` | `40` | `\n` | `\n` | `\n` | **FAIL** |
| **Arithmetic** | `12 * 8 = ` | `96` | `\n` | `\n` | `\n` | **FAIL** |
| **Arithmetic** | `100 - 35 = ` | `65` | `\n` | `\n` | `\n` | **FAIL** |
| **Arithmetic** | `81 / 9 = ` | `9` | `\n` | `\n` | `\n` | **FAIL** |
| **Multi-Step** | `Calculate: (10 + 5) * 4 = ` | `60` | `\n` | `\n` | `\n` | **FAIL** |
| **Multi-Step** | `First add 20 and 30, then multiply by 2: ` | `100` | `\n` | `\n` | `\n` | **FAIL** |
| **Code Completion** | `def multiply(x, y):\n    return ` | `x * y` | `\n` | `None` | `None` | **FAIL** |
| **Code Completion** | `class Queue:\n    def __init__(self):\n        self.items = ` | `[]` | `\n` | `[` | `[` | **PASS (1/13)** |
| **Code Completion** | `import json\ndata = json.loads(` | `text)` | `\n` | `f` | `f` | **FAIL** |
| **Propositions** | `ChakrMicro unique trainable parameter count is ` | `3443136` | `\n` | `1` | `1` | **FAIL** |
| **Propositions** | `The safe calculator strictly forbids python eval and ` | `exec` | `\n` | `the` | `the` | **FAIL** |
| **Hinglish** | `ChakrView model local hardware par bina API ke ` | `run` | `\n` | `bhi` | `bhi` | **FAIL** |
| **Hinglish** | `Python sandbox ke andar forbidden modules ko ` | `block` | `\n` | `bhi` | `bhi` | **FAIL** |

---

### 3. Key Observations & Invalidation of Capability Claims

1. **Failure on Arithmetic and Multi-Step Math:**
   The neural core predicts newline tokens `\n` for all mathematical prompts, completely failing basic single-digit and multi-digit arithmetic. This demonstrates why ChakrView's architecture mandates the `SafeCalculator` tool rather than relying on neural token generation.
2. **Failure on Proposition Recall:**
   Despite extensive pretraining on system architecture files mentioning parameter counts (3,443,136), the adapted model outputs `1` rather than the true parameter count.
3. **Absence of Scaling in Transfer:**
   Extending training from 150 steps to 300 steps (Condition D) yielded **identical 1/13 transfer performance**, confirming that the 300-step model suffered from severe template memorization without gaining any generalizable functional capability.
