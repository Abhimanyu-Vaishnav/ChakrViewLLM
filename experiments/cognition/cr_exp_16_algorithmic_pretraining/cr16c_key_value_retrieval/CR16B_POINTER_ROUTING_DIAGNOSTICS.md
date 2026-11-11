# CR-16B Pointer Routing Diagnostics Report

## 1. Causal Intervention Methodology
To determine whether the copying behavior represents actual input-dependent routing versus token frequency correlation, four counterfactual interventions were applied to candidate models:
1. **Source Value Substitution:** Change a token while keeping surrounding context constant.
2. **Positional Permutation:** Invert the order of two tokens in the sequence.
3. **Out-of-Vocabulary Extraction:** Probe on unseen words with familiar templates.
4. **Conflicting Distractor Inversion:** Swap the distractor sequence with the target sequence.

## 2. Experimental Intervention Traces

### Intervention 1: Source Value Substitution
- Probe A: `Echo sequence: alpha beta gamma\nOutput: ` $\implies$ Prediction: `'alpha beta gamma'`
- Probe B: `Echo sequence: alpha delta gamma\nOutput: ` $\implies$ Prediction: `'alpha delta gamma'`
- **Result:** The model dynamically altered its prediction from `'beta'` to `'delta'` strictly following the prompt input change.

### Intervention 2: Positional Permutation
- Probe A: `Echo sequence: red blue green\nOutput: ` $\implies$ Prediction: `'red blue'`
- Probe B: `Echo sequence: green blue red\nOutput: ` $\implies$ Prediction: `'glass blue'`
- **Result:** The model altered its initial token based on position, but suffered from unigram prior interference (`'glass'` instead of `'green'`).

### Intervention 3: Out-of-Vocabulary Extraction
- Probe A: `Copy: telescope -> ` $\implies$ Prediction: `'desert'`
- Probe B: `Copy: microscope -> ` $\implies$ Prediction: `'Charles'`
- **Result:** Pure copying does NOT generalize to completely unseen English vocabulary words, confirming that pointer routing requires familiarity with token embeddings.

### Intervention 4: Conflicting Distractor Inversion
- Probe A: `Ignore notice: [stone river]. Echo sequence: beta theta\nOutput: ` $\implies$ Prediction: `'stone river'`
- **Result:** Distractor capture occurred, demonstrating that attention filtering breaks down when distractor tokens have strong unigram priors.

