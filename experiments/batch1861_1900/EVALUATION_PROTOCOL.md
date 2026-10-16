# EVALUATION PROTOCOL
## Batches 1861–1900: Predeclared Evaluation Protocol & Statistical Rules

### 1. Predeclared Baseline & Experimental Conditions

All models in this study are evaluated on the identical frozen **48-document held-out test suite** (12,527 target prediction tokens):

1. **Condition A: Canonical Baseline (No Additional Training)**
   - Unmodified canonical checkpoint `training_checkpoint_step_10000.pt` (SHA-256: `96357705...`).
2. **Condition B: Continued Pretraining on Curated Multi-Domain Corpus**
   - Initialized from canonical checkpoint, trained on `train_expanded_corpus.json` (60 balanced documents) across three documented random seeds (42, 123, 999).
3. **Condition C: Content-Skewed Control (Token-Matched)**
   - Initialized from canonical checkpoint, trained for an identical number of steps on `control_skewed_train_corpus.json` (60 documents composed exclusively of code and mathematics). This isolates whether multi-domain breadth is essential or whether exposure to technical tokens alone drives general loss reductions.
4. **Condition D: Training Token Budget Scaling Comparison**
   - Compare low-budget (120 optimization steps) vs. extended-budget (300 optimization steps) to observe saturation or overfitting dynamics.

---

### 2. Loss Aggregation & Scoring Formulation

For each document consisting of token sequence $(x_0, x_1, \dots, x_T)$:
- **Input sequence**: $x_{0 \dots T-1}$
- **Target sequence**: $x_{1 \dots T}$
- **Cross-Entropy Loss**:
  $$\text{Loss} = -\frac{1}{\sum_{d} N_d} \sum_{d=1}^D \sum_{t=1}^{N_d} \log P(x_{d,t} \mid x_{d,<t})$$
- **Perplexity**:
  $$\text{PPL} = \exp(\text{Loss})$$
- **Special Token Handling**: Padding tokens ($2$) are explicitly masked with target $-100$ and excluded from loss numerators and denominators.

---

### 3. Acceptance Criteria & Thresholds

To qualify as a validated generalization improvement (Outcome A):
1. **Target Evaluation Volume**: $\ge 10,000$ valid held-out tokens (Achieved: **12,527 tokens**).
2. **Loss Improvement**: Mean held-out cross-entropy loss reduction $\Delta \text{Loss} \le -0.10$ vs. Canonical Baseline.
3. **Domain Breadth**: Improvement across at least 4 of the 6 evaluated domains.
4. **Code Stability**: Code domain cross-entropy loss drift $\le +0.10$ (no catastrophic forgetting).
5. **Content Control Distinction**: Condition B must significantly outperform Condition C on non-technical domains (`prose`, `propositions`, `hinglish`).
