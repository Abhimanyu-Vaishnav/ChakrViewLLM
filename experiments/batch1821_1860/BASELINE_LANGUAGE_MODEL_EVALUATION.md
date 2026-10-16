# BASELINE LANGUAGE MODEL EVALUATION REPORT
## Batches 1821–1860: Canonical ChakrMicro Step 10,000 Next-Token Performance

### 1. Evaluation Methodology & Metrics Definition

The canonical baseline evaluation measures the intrinsic language modeling capabilities of `ChakrMicro` (Step 10,000, SHA-256: `96357705...`) across a newly frozen, document-level held-out corpus spanning six distinct domains:
- **Corpus File**: `experiments/batch1821_1860/data/test_held_out_corpus.json`
- **Total Held-Out Documents**: 18 documents (3 per domain across `prose`, `technical`, `mathematics`, `code`, `propositions`, `hinglish`).
- **Total Evaluated Targets**: 708 next-token predictions.
- **Evaluation Objective**: Autoregressive causal cross-entropy loss over ground truth tokens:
  $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}), \quad \text{Perplexity} = \exp(\text{Loss})$$
- **Top-1 / Top-5 Accuracy**: Exact match between the top ranked logit and ground truth next-token, and presence of target within top 5 candidates.

---

### 2. Canonical Baseline Performance Matrix

| Evaluation Domain | Evaluated Tokens | Cross-Entropy Loss | Perplexity (PPL) | Top-1 Next-Token Accuracy | Top-5 Next-Token Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **General Prose** | 111 | 5.4468 | 232.01 | 12.61% | 26.13% |
| **Technical Documentation** | 109 | 5.9977 | 402.50 | 11.01% | 27.52% |
| **Mathematics & Formulae** | 131 | 5.3443 | 209.40 | 9.92% | 26.72% |
| **Python Code** | 175 | **3.8286** | **46.00** | **28.57%** | **48.00%** |
| **Factual Propositions** | 88 | 5.5794 | 264.90 | 11.36% | 29.55% |
| **Hinglish Conversational** | 94 | 7.1692 | 1298.76 | 5.32% | 17.02% |
| **Overall Held-Out Baseline** | **708** | **5.3578** | **212.26** | **14.69%** | **31.07%** |

---

### 3. Key Baseline Observations

1. **Code is the Strongest Inherent Domain**:
   - Python code achieves the lowest loss (**3.8286**) and lowest perplexity (**46.00**), with **28.57% Top-1 accuracy** and **48.00% Top-5 accuracy**.
   - The canonical pre-training corpus imparted strong indentation and keyword syntax structure (`def`, `return`, `for`, `in`).
2. **Hinglish Exhibits Severe Subword Fragmentation Penalty**:
   - Hinglish has the highest loss (**7.1692**) and perplexity (**1298.76**).
   - Because the tokenizer lacks Hindi Romanized subword merges, Hinglish words are fragmented into single-character or short subwords, inflating next-token uncertainty.
3. **Transfer Task Performance**:
   - On the 8 diagnostic continuation tasks (arithmetic, code completion, propositions), the canonical model achieved **0.0% exact transfer accuracy**, generating generic punctuation (`"<"`, `"\""`) rather than completing direct arithmetic answers.
