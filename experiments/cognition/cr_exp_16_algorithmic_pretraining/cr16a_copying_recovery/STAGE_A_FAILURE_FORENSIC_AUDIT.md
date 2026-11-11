# Stage A Failure Forensic Audit

## 1. Context of the Failure
In CR-EXP-16, the Stage A pilot was intended to establish foundational sequence copying on ChakrMicro before conversational fine-tuning. The pilot reported:
- Training steps: 150
- Batch size: 16
- Optimizer: AdamW ($lr=1\times 10^{-4}$)
- Initial validation loss: $4.6302$
- Final validation loss: $1.8828$
- Evaluation result on 300 frozen copying probes: **0 / 300 Exact Match (0.00%)**, **0 / 300 Token Presence (0.00%)**.

## 2. Forensic Reconstruction & Token-Level Tracing
We analyzed the pilot training data (`stage_a_train.json`) and checkpoint (`stage_a_pilot_final.pt`, SHA-256: `e2280978...`).

### A. Sequence Formatting
Samples were formatted as:
```text
Input: banana one\nCopy: banana one\n
Sequence: two yellow rain orange lake -> two yellow rain orange lake\n
```

### B. Loss Computation Mismatch
The training script computed cross-entropy loss over every token in the concatenated sequence:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})$$
For a sample of length $N = 12$:
- Prompt tokens: `['Input', ':', ' banana', ' one', '\n', 'Copy', ':']` (7 tokens)
- Target tokens: `[' banana', ' one', '\n']` (3 tokens)
- Prompt formatting tokens accounted for $>60\%$ of total tokens in each batch.

### C. The Unigram Fitting Pathology
Because the network is small ($3.44\text{M}$ parameters) and trained from a Stage B conversational checkpoint where language priors already exist, the fastest way for gradient descent to reduce loss without forming cross-attention induction heads is:
1. Memorize the syntax of the prompt prefixes (`Input:`, `Copy:`, `Sequence:`).
2. Predict the empirical unigram marginal distribution of the corpus (`stone`, `storm`, `blue`, `north`).

When evaluated in free generation, the Stage A pilot outputted:
- Input: `Echo sequence: delta beta omicron zeta\nOutput: `
- Model Output: `north stone stone stone ston`
- Input: `Echo sequence: xi kappa zeta pi\nOutput: `
- Model Output: `north stone stone stone ston`

The loss decrease from $4.63 \to 1.88$ was entirely driven by memorizing prompt templates and unigram statistics, while copying accuracy remained exactly $0.0\%$.

