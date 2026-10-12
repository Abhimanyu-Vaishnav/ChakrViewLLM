# Batch 86 — Synthesis, Audit, and Long-Term Research Recommendations

## 1. Independent Audit of Sprints 71–86

Across 16 consecutive sprint batches (71 through 86), ChakrMicro v0.1 (3,443,136 parameters, 6 layers, $d_{model}=192$) was rigorously evaluated under multiple training regimes to assess mathematical generalization on single-digit addition:

| Sprint Batches | Experimental Regime | Held-Out Test Generalization | Observed Failure Mode / Mechanism |
|:---|:---|:---:|:---|
| **Batches 71–74** | Uniform & Answer-Emphasis SFT | 0.0% | Collapsed to single constant modal digit `7`. |
| **Batches 75–78** | Multi-Template Synthetic Pretraining | 0.0% | Template loss dropped to 0.0016; collapsed to constant digit `6`. |
| **Batches 79–82** | Uniform Sum-Class Balancing ($0 \dots 18$) | 5.56% | Broken into piecewise threshold heuristic on first operand $a$. |
| **Batches 83–86** | Intermediate Counting Scratchpad | 0.0% | Formatted outputs reached 100% valid syntax, but memorized canned counting sequences (`Start at 0...` or `Start at 9...`). |

---

## 2. Key Scientific Conclusions

1. **Format and Template Competence is Disconnected from Functional Logic**:
   - In all conditions, ChakrMicro demonstrated rapid mastery of prompt templates, syntax delimiters, and EOS generation (100% adherence).
   - Low loss on template tokens is not evidence of reasoning.
2. **The Causal Attention Bottleneck in Small Transformers**:
   - Small models (3.44M params, 6 layers) optimize cross-entropy by finding statistical shortcuts (first-token thresholding or memorizing canonical template strings) rather than learning dynamic composition of multiple input tokens.
   - Simply expanding the sequence with intermediate scratchpad tokens allows the model to memorize canned scratchpad examples unless structural inductive biases or curriculum constraints enforce step-by-step operand dependency.

---

## 3. Recommended Next Engineering Interventions

1. **Step-by-Step Masked Autoregressive Scaffolding**:
   - Train on scratchpad sequences where the loss is applied only to the next step given the previous step, penalizing template reuse.
2. **Explicit Positional & Operator Embeddings**:
   - Explore number representation embeddings where magnitude is encoded directly into embedding space.
3. **Expand Workbench Product Features**:
   - Integrate evaluation dataset management and comparative model diffing directly into the local workbench UI.
