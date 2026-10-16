# GO / NO-GO DECISION
## Batches 1781–1820: Final Evaluation of Neural Advantage & Acceptance Criteria

### 1. Predeclared Acceptance Criteria Evaluation

Before evaluating the final experimental models, the acceptance criteria were frozen according to Work Package 6:

| # | Acceptance Criterion | Predeclared Threshold | Best Experimental Candidate Result | Verdict |
| :-: | :--- | :---: | :---: | :---: |
| **1** | **Held-out intent accuracy** | $\ge 70.0\%$ | **55.89% ± 1.84%** (Variant B) / **42.45%** (Intervention A) | ❌ **FAIL** |
| **2** | **Macro-F1 score** | $\ge 0.700$ | **0.5280 ± 0.0195** (Variant B) / **0.3809** (Intervention A) | ❌ **FAIL** |
| **3** | **Meaningful advantage over strongest baseline** | $> +5.0\%$ over baseline | **+0.56% over Keyword** (55.89% vs 55.33%)<br>**+3.89% over TF-IDF** (55.89% vs 52.00%) | ❌ **FAIL** |
| **4** | **No catastrophic adversarial failures** | $\le 10.0\%$ false positive rate | **100.0% FP** (Variant B) / **20.0% FP** (Intervention A) | ❌ **FAIL** |
| **5** | **Separately measured argument accuracy** | Documented | Exact Match: 52.00% (Variant B) | ✅ **PASS** |
| **6** | **No regression in governed authorization** | No regressions | Preserved via Governed Firewall | ✅ **PASS** |
| **7** | **Multi-seed consistency** | Reasonable across 3 seeds | Std Dev: 1.84% (Variant B), 2.68% (Int A) | ✅ **PASS** |

---

### 2. Scientific Diagnosis of the Learning Bottleneck

1. **Why Does Accuracy Plateau at ~56%?**
   ChakrMicro Step 10,000 is a compact 12.5M parameter transformer pre-trained on foundational language objectives. When fine-tuned on a 240-example intent dataset, it rapidly memorizes lexical associations, but its internal representations lack the compositional depth to distinguish fine-grained operational intents across genuinely unseen syntactic families.
2. **Why Do Interventions Fail to Beat the Baseline?**
   - **Intervention A (Adversarial Awareness)**: Dropped accuracy from 55.89% to 42.45% because the small model over-generalized the negative association, classifying legitimate code and math commands as `OUT_OF_SCOPE`.
   - **Intervention B (Contrastive Margin)**: Dropped accuracy to 40.11% due to feature space collapse on limited training samples.
   - **Intervention C (Attention Pooling)**: Dropped accuracy to 34.22% due to attention query overfitting on punctuation and filler words.
3. **Keyword Heuristic Equivalence**:
   A 50-line regex heuristic achieves **55.33%** intent accuracy. The fine-tuned transformer achieves **55.89%**. A 12.5M parameter neural network that outperforms a simple keyword script by only +0.56% does not demonstrate meaningful neural routing capability.

---

### 3. Formal Final Decision

According to the mission instructions, exactly one option must be selected:

> **A. MEASURABLE NEURAL ADVANTAGE DEMONSTRATED** — the candidate clears the frozen evaluation gates and meaningfully outperforms simple baselines.  
> **B. NO RELIABLE NEURAL ADVANTAGE YET** — results are insufficient, inconsistent, or below the gates.

### **FINAL DECISION: B. NO RELIABLE NEURAL ADVANTAGE YET**

The experimental candidates fail the accuracy gate (55.89% vs $\ge 70\%$), fail the macro-F1 gate (0.5280 vs $\ge 0.70$), fail to demonstrate a meaningful margin over simple baselines (+0.56% over keyword, +3.89% over TF-IDF), and exhibit severe adversarial routing vulnerabilities.

The direct neural intent classifier **MUST NOT** be integrated into the production runtime.

---

### 4. Single Most Informative Next Experiment

The results conclusively demonstrate that optimizing classification heads, regularizing losses, or augmenting small synthetic datasets will not overcome the 56% accuracy ceiling.

**Recommended Next Step**:
The single most informative experiment is a **Task-Grounded Pre-Training / Masked Span Alignment Milestone**:
Pre-train the ChakrMicro transformer backbone on a large corpus of structured code, mathematical expressions, and technical propositions using masked span reconstruction before attaching any routing or classification heads. True neural generalization requires representation-level semantic grounding, not superficial linear probing on compact weights.
