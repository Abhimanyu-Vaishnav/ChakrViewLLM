# Label Alignment Audit

## 1. Autoregressive Shift Mechanics
In decoder-only autoregressive Transformers, the forward pass predicts the next token at every position:
$$\hat{y}_t = \text{argmax } P(x_{t+1} \mid x_1, \dots, x_t)$$
Therefore, the alignment between input IDs and target labels requires an exact one-position right shift:
- Inputs: `input_ids[:, :-1]`
- Targets: `labels[:, 1:]`

## 2. Forensic Audit of Stage A Pilot Alignment
In the historical Stage A pilot, the full sequence was fed as:
- Input: `['Input', ':', ' banana', ' one', '\n', 'Copy', ':', ' banana', ' one', '\n']`
- Targets: `[':', ' banana', ' one', '\n', 'Copy', ':', ' banana', ' one', '\n', '<EOS>']`

Under this formulation:
- At position 0 (`Input`), the model is penalized if it doesn't predict `:`.
- At position 1 (`:`), the model is penalized if it doesn't predict the space/first word.
- At position 6 (`:`), the model is penalized if it doesn't predict the copied first word.

Because the loss function averages across all 10 positions, the gradient signal is dominated by syntax tokens that appear identically across every training example. The model minimizes loss by lowering the entropy on predictable syntax rather than developing attention routing from position 6 back to position 2.

## 3. Corrected Target-Masked Formulation
In CR-EXP-16A, target masking was implemented by setting all prompt token labels to the PyTorch ignore index (`-100`):
- `labels = [-100] * len(prompt_ids) + list(target_ids)`
- After shift:
  - At position `len(prompt_ids) - 1` (the last prompt delimiter, e.g., `'Output: '`):
    - The target label is `target_ids[0]` (the first copied token).
    - Loss is actively computed: $\mathcal{L}_{\text{first}} = -\log P(\text{target}_0 \mid \text{prompt})$.
  - For all positions $t < \text{len(prompt\_ids)} - 1$:
    - The target label is `-100`.
    - Cross-entropy loss ignores these positions completely: $\text{Weight} = 0$.

This mathematically guarantees that 100% of the parameter updates contribute exclusively to the conditional probability of the target given the prompt context.

