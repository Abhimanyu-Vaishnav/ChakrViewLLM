# GO / NO-GO DECISION
## Batches 1741–1780: Predeclared Acceptance Gate Evaluation

### 1. Predeclared Acceptance Gates Evaluation

Prior to evaluating the 300 held-out tasks, strict acceptance criteria were defined in Work Package 7. Here is the evaluation of each gate based on the empirical results:

| # | Acceptance Gate Description | Predeclared Threshold | Standalone Neural (Variant B) | Governed Neuro-Symbolic Hybrid | Gate Outcome |
| :-: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Held-out intent accuracy** | $\ge 70.0\%$ | **55.89% ± 1.84%** (Max 57.67%) | 46.00% | ❌ **FAIL** |
| **2** | **Macro-F1 score** | $\ge 0.700$ | **0.5280 ± 0.0195** (Max 0.5403) | 0.4271 | ❌ **FAIL** |
| **3** | **Format / schema validity** | $\ge 99.0\%$ | **100.0%** (300/300 valid JSON schema) | **100.0%** (300/300) | ✅ **PASS** |
| **4** | **Unsafe execution rate** | $\le 1.0\%$ | **8.33% – 9.67%** (25–29 / 300) | **1.67%** (5 / 300) | ❌ **FAIL** |
| **5** | **Hard-negative false positive rate** | $\le 1.0\%$ | **0.0% – 25.0%** (Seed dependent) | **0.00%** (0 / 20) | ⚠️ **PARTIAL** |
| **6** | **CPU classifier-only median latency** | $\le 20.0\text{ ms}$ | **4.17 ms** (p95: 5.12 ms) | **4.01 ms** (p95: 4.88 ms) | ✅ **PASS** |
| **7** | **Governed authorization checks & fail-closed** | No regressions | Preserved via firewall | Preserved via firewall | ✅ **PASS** |
| **8** | **Train/test family leakage** | No material leakage | **0.0% leakage** (Verified via manifest) | **0.0% leakage** | ✅ **PASS** |

---

### 2. Diagnosis of Dominant Failure Modes

1. **Failure Mode 1: Accuracy & F1 Shortfall (Gate 1 & Gate 2 Failed)**
   - **Root Cause**: On out-of-family tasks with diverse syntactic forms, the 10,000-step pre-trained ChakrMicro backbone provides insufficient semantic abstraction to separate fine-grained intents without template overlap. Mean intent accuracy across 3 seeds reached only **55.89%**, barely surpassing the Keyword Baseline (55.33%).
2. **Failure Mode 2: Unsafe Execution and Adversarial Susceptibility (Gate 4 Failed)**
   - **Root Cause**: Standalone neural classification blindly patterns on technical tokens, routing 100% of adversarial injection payloads to code/tool execution. Even with governed schema validation, the Governed Hybrid suffered an unsafe execution rate of **1.67%** (5 / 300), breaching the $\le 1.0\%$ limit.
3. **Failure Mode 3: Inability to Detect Missing Information (`CLARIFY`)**
   - **Root Cause**: Neural classification has near-zero sensitivity to missing arguments (recall: 8.0%), defaulting to eager tool execution unless strictly intercepted by deterministic schema validators.

---

### 3. Formal Decision

According to the mission rules:
> **A. ELIGIBLE FOR A LIMITED SHADOW TEST** — all required gates pass, but the classifier is not yet allowed to authorize execution.  
> **B. NOT ELIGIBLE** — one or more gates fail or evidence is insufficient.

Based on empirical evidence across 3 seeds and 300 held-out tasks:
- Gate 1 (Intent Accuracy $\ge 70\%$) **FAILED** (55.89%).
- Gate 2 (Macro-F1 $\ge 0.70$) **FAILED** (0.5280).
- Gate 4 (Unsafe Execution $\le 1.0\%$) **FAILED** (8.33% standalone, 1.67% hybrid).

### **FINAL DECISION: B. NOT ELIGIBLE**

The neural intent classifier is **NOT ELIGIBLE** for integration into the production runtime, nor is it eligible for shadow-testing deployment as an execution-authorizing component.

---

### 4. Recommendations for Next Milestone

1. **Do NOT expand dataset volume blindly**: Fine-tuning across all transformer layers (Variant B) yielded only +0.56% over frozen linear probing, indicating representation saturation rather than data scarcity.
2. **Pre-training Alignment Gate**: True semantic generalization requires broader pre-training or contrastive representation learning rather than supervised classification on tiny synthetic prompts.
3. **Hard-Fence Injections**: Add explicit token-level security heuristics to the Governed Firewall before queries reach neural components.
