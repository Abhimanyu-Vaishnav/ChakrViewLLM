# Batch 165 — Neural Copy-Induction Loss Contract & Target Masking
## Causal Masking, Autoregressive Boundary Alignment & Loss Invariants

### 1. The Core Invariant
In autoregressive language modeling, standard causal SFT trains across all tokens:
$$\mathcal{L}_{\text{total}} = -\sum_{t=1}^T \log P(x_t \mid x_{<t})$$
This objective severely dilutes the gradient for in-context factual copying: the model spends parameter capacity predicting formulaic prompt framing tokens ("Question:", "Context:", etc.) rather than learning to copy the requested entity from the context span.

### 2. The Copy-Induction Loss Mask Formulation
Under the copy-induction loss contract:
1. **Prompt & Context Tokens**: Masked to `IGNORE_INDEX` (-100). Zero cross-entropy loss gradients are backpropagated through user questions or context framing.
2. **Answer Target Tokens**: Retained with active cross-entropy loss.
3. **Turn Termination Token (`<|im_end|>` / EOS)**: Retained with active cross-entropy loss to enforce clean stopping without infinite repetition loops.
4. **Causal Shift**: Target sequence $Y$ is shifted by one position relative to input $X$:
   - Inputs: $x_0, x_1, \dots, x_{T-1}$
   - Targets: $y_1, y_2, \dots, y_T$
   - Mask: $M_t = 1$ if $y_t \in \text{Answer} \cup \{\text{EOS}\}$ else $-100$.

### 3. Masking Verification Fixture
```
Input text: "<|im_start|>user\nContext: key is val.\nQuestion: What is key?<|im_end|>\n<|im_start|>assistant\nval<|im_end|>"

Tokens:
  [Prompt / Context]:  <|im_start|> ... assistant\n       -> Target label = -100
  [Answer Tokens]:     val                                -> Target label = ID('val')
  [Termination Token]: <|im_end|>                         -> Target label = ID('<|im_end|>')
```
Loss is strictly calculated only where the model must generate the answer and stop.
