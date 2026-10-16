# GENERALIZATION AND TRANSFER REPORT
## Batches 1821–1860: Out-of-Distribution Generalization & Practical Task Transfer

### 1. Generalization vs. Memorization Verification

To establish whether the -0.2317 loss reduction reflects genuine distribution generalization rather than training set memorization:

1. **Document-Level Partitioning**: All 18 held-out test documents were excluded entirely from training batches.
2. **Per-Document Delta Distribution**:
   - Out of the 18 held-out documents, **15 showed lower loss** under the continued pre-training model than under the canonical baseline.
   - For example:
     - `doc_technical_018` (Operating systems / cgroups): Loss improved from **7.0348 to 6.6335** ($\Delta = -0.4013$).
     - `doc_hinglish_058` (ChakrView hardware efficiency): Loss improved from **8.2595 to 7.4211** ($\Delta = -0.8384$).
     - `doc_prose_010` (Scientific reproducibility): Loss improved from **5.3517 to 4.9685** ($\Delta = -0.3832$).
3. **Absence of Catastrophic Forgetting**:
   - Code syntax loss remained robust at 3.86 (PPL 47.7), demonstrating that the weights did not over-specialize on prose at the expense of previously mastered programming token distributions.

---

### 2. Diagnostic Transfer Tasks Evaluation

To rigorously test whether language-model perplexity reductions transfer to practical operational tasks, a frozen suite of 8 diagnostic tasks was evaluated:

| Task Category | Prompt | Expected Answer | Canonical Output (First Token) | CPT Output (First Token) | Practical Transfer Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Arithmetic** | `Calculate: 15 * 4 = ` | `60` | `"<"` | `"<"` | ❌ No Transfer |
| **Arithmetic** | `Calculate: 99 - 33 = ` | `66` | `"<"` | `"<"` | ❌ No Transfer |
| **Arithmetic** | `Calculate: 50 + 25 = ` | `75` | `"<"` | `"<"` | ❌ No Transfer |
| **Code Completion** | `def add(a, b):\n    return ` | `a + b` | `"p"` | `"a"` | ⚠️ Improved Prefix |
| **Code Completion** | `for i in range(10):\n    print(` | `i)` | `"\""` | `"\""` | ❌ No Transfer |
| **Propositions** | `ChakrMicro unique trainable parameter count is ` | `3443136` | `"a"` | `"3"` | ⚠️ Improved Prefix (`"3"`) |
| **Propositions** | `The safe calculator forbids python eval and ` | `exec` | `"the"` | `"the"` | ❌ No Transfer |
| **Hinglish** | `ChakrView model local hardware par ` | `run` | `"is"` | `"run"` | ✅ Exact Transfer Match |

#### Overall Transfer Task Score:
- **Canonical Model Transfer Accuracy**: **0 / 8 (0.0%)**
- **Continued Pre-Training Transfer Accuracy**: **1 / 8 (12.5%)** (plus 2 partial prefix improvements on `add` and parameter count).

---

### 3. Scientific Implications: What Perplexity Reduction Proves vs. What It Does Not

> **Crucial Scientific Distinction**:
> A reduction in causal cross-entropy loss and held-out perplexity proves that the model has learned the statistical n-gram transitions and domain vocabulary of the target distribution.
> 
> **It DOES NOT prove that the model has learned mathematical computation, code logic, or universal factual reasoning.**
> 
> When queried with `"Calculate: 15 * 4 = "`, both models generate generic HTML/token tags (`"<"`) rather than computing arithmetic. The model remains a miniature 3.44M language model: it predicts token likelihoods, not arithmetic truth. Arithmetic correctness and execution security must remain governed by deterministic tools (`SafeCalculator`).
