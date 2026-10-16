# PRACTICAL TRANSFER EVALUATION REPORT
## Batches 1861–1900: Diagnostic Transfer Suite & Boundaries of Autoregressive Prediction

### 1. Diagnostic Transfer Suite Specification

To avoid conflating language modeling perplexity reductions with operational intelligence or task reasoning, we evaluated an expanded frozen suite of **13 diagnostic transfer tasks** across four functional categories:
1. **Single-Step Arithmetic** (Addition, subtraction, multiplication, division).
2. **Multi-Step Arithmetic** (Compound expressions with explicit operations).
3. **Syntax-Sensitive Code Completion** (Functions, classes, imports).
4. **Propositions & Policy Continuer** (Parameter counts, security constraints).
5. **Hinglish Conversational Continuation** (Hardware efficiency, sandbox policies).

---

### 2. Comparative Transfer Task Results

| Task Category | Prompt String | Ground Truth Target | Canonical Output | Condition B Output (150 Steps) | Condition D Output (300 Steps) | Task Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arithmetic** | `15 + 25 = ` | `40` | `"<"` | `"<"` | `"$"` | ❌ No calculation |
| **Arithmetic** | `12 * 8 = ` | `96` | `"<"` | `"<"` | `"<"` | ❌ No calculation |
| **Arithmetic** | `100 - 35 = ` | `65` | `"<"` | `"<"` | `"<"` | ❌ No calculation |
| **Arithmetic** | `81 / 9 = ` | `9` | `"<"` | `"<"` | `"<"` | ❌ No calculation |
| **Multi-Step** | `Calculate: (10 + 5) * 4 = ` | `60` | `"<"` | `"<"` | `"<"` | ❌ No calculation |
| **Multi-Step** | `First add 20 and 30, then multiply by 2: `| `100` | `"The"` | `"2"` | `"The"` | ❌ No calculation |
| **Code Completion** | `def multiply(x, y):\n return ` | `x * y` | `"p"` | `"n"` | `"x"` | ⚠️ Improved Prefix (`"x"`) |
| **Code Completion** | `class Queue:\n def __init__(self):\n self.items = ` | `[]` | `"self."` | `"self."` | `"{}"` | ⚠️ Valid Syntax (`"{}"`) |
| **Code Completion** | `import json\ndata = json.loads(` | `text)` | `""` | `""` | `"\""` | ❌ Generic quote |
| **Propositions** | `ChakrMicro unique trainable parameter count is ` | `3443136` | `"a"` | `"a"` | `"3"` | ⚠️ Improved Prefix (`"3"`) |
| **Propositions** | `The safe calculator strictly forbids python eval and ` | `exec` | `"the"` | `"the"` | `"exec"` | ✅ Exact Continuation |
| **Hinglish** | `ChakrView model local hardware par bina API ke ` | `run` | `"the"` | `"the"` | `"run"` | ✅ Exact Continuation |
| **Hinglish** | `Python sandbox ke andar forbidden modules ko ` | `block` | `"the"` | `"the"` | `"block"` | ✅ Exact Continuation |

#### Transfer Accuracy Summary:
- **Canonical Step 10,000**: **0 / 13 (0.0%)** passed.
- **Condition B (150 Steps)**: **0 / 13 (0.0%)** passed.
- **Condition D (300 Steps)**: **3 / 13 (23.1%)** passed (with 3 additional prefix improvements).

---

### 3. Critical Scientific Distinction: Calculation vs. Statistical Completion

> **Fundamental Architectural Finding**:
> 1. Even when mathematical derivation perplexity drops to **27.7** in Condition D, **the neural core does NOT perform mental arithmetic**. It produces formatting tokens (`"<"`, `"$"`), failing to output `"40"` or `"96"`.
> 2. What the model *does* learn is **vocabulary-level syntactic completion**:
>    - Successfully completes `"The safe calculator strictly forbids python eval and "` $\to$ `"exec"`.
>    - Successfully completes Hinglish colloquial verbs (`"run"`, `"block"`).
> 3. **Governance Mandate**: Mathematical calculations and code execution MUST remain strictly delegated to deterministic tools (`SafeCalculator`, `SandboxedPythonTool`). The neural model's role in ChakrView is linguistic representation, not unverified autonomous computation.
