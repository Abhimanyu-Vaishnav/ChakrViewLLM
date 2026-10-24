# ChakrView Exp17 Instruction SFT v1 — Training Protocol Specification

**Experiment ID**: `exp17_instruction_sft_v1`  
**Parent Directory**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Base Engine**: `chakrview/training/sft_engine.py` (Stage D SFT Engine)  
**Status**: TRAINING PROTOCOL SPECIFICATION ONLY

---

## 1. Mathematical Objective & Loss Masking Formulation

Exp17 trains the ChakrMicro autoregressive causal language model using **Prompt-Masked Cross-Entropy Loss**.

### 1.1 Sequence Alignment
Given a prompt token sequence $\mathbf{p} = [p_1, p_2, \dots, p_L]$ and a target response token sequence $\mathbf{r} = [r_1, r_2, \dots, r_M]$, the training input and target sequences are constructed via `build_causal_sequence()` ([prompt_contract.py:54-56](file:///D:/Project/ChakrView/chakrview/training/prompt_contract.py#L54-L56)):

$$\mathbf{x} = [\text{BOS}, p_1, p_2, \dots, p_L, r_1, r_2, \dots, r_M]$$
$$\mathbf{y} = [\underbrace{-100, -100, \dots, -100}_{L \text{ prompt tokens}}, r_1, r_2, \dots, r_M, \text{EOS}]$$

- Total sequence length: $T = 1 + L + M$.
- At position $t = 0$ (token $\text{BOS}$), target is $-100$ (no loss computed).
- At position $t = L$ (the final prompt token $p_L$), the model's forward prediction is aligned with target $y_L = r_1$ (the first response token).
- At position $t = L + M$ (the final response token $r_M$), the model's forward prediction is aligned with target $y_{L+M} = \text{EOS}$ (token ID `1`).

### 1.2 Loss Function
The empirical loss for sequence $i$ over active response positions $\mathcal{A}_i = \{t \mid y_{i,t} \ne -100\}$ is:

$$\mathcal{L}_i(\theta) = - \frac{1}{|\mathcal{A}_i|} \sum_{t \in \mathcal{A}_i} \log P_\theta(y_{i,t} \mid x_{i, \le t})$$

The batch loss across batch size $B$ is the token-weighted mean:
$$\mathcal{L}_{\text{batch}}(\theta) = \frac{\sum_{i=1}^B \sum_{t \in \mathcal{A}_i} \ell(x_{i,t}, y_{i,t})}{\sum_{i=1}^B |\mathcal{A}_i|}$$

This guarantees that:
1. Gradients flow exclusively through response generation and turn termination ($\text{EOS}$).
2. The model receives zero gradient penalty for the prompt structure or user query.
3. $\text{EOS}$ emission is directly rewarded as the mandatory terminus of every response turn.

---

## 2. Hyperparameter Configuration & Optimization

All hyperparameters are selected to ensure stable fine-tuning of the 3.44M-parameter checkpoint on CPU without catastrophic forgetting:

| Hyperparameter | Value | Rationale & Invariant |
| :--- | :--- | :--- |
| **Base Checkpoint** | `step_10000.pt` | Canonical pretrained checkpoint (SHA-256: `96357705...`). |
| **Optimizer** | AdamW (`torch.optim.AdamW`) | Standard decoupled weight decay optimizer. |
| **Peak Learning Rate ($\eta_{\text{max}}$)** | $5.0 \times 10^{-5}$ | Conservative peak LR; prevents destruction of pretrained representations. |
| **Min Learning Rate ($\eta_{\text{min}}$)** | $5.0 \times 10^{-6}$ | 10% of peak learning rate at schedule termination. |
| **Learning Rate Schedule** | Cosine Annealing | Smooth decay: $\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\pi t / T_{\text{total}}))$. |
| **Weight Decay** | $0.01$ | Mild regularization on non-norm weights. |
| **Gradient Clipping** | $1.0$ (`max_norm`) | Prevents gradient explosion on outlier sequences. |
| **Batch Size ($B$)** | $2$ sequences | Optimal cache/memory trade-off for CPU execution. |
| **Total Optimizer Steps ($T_{\text{total}}$)** | $400$ steps | $\approx 2.5$ epochs over 320 training examples ($320 / 2 = 160$ steps/epoch). |
| **Sequence Length Ceiling** | $512$ tokens | Enforces `ModelConfig.max_seq_len = 512`. |
| **Validation Interval** | Every $25$ steps | Tracks validation loss and perplexity on 40 held-out validation samples. |
| **Checkpoint Save Interval** | Every $100$ steps | Periodic atomic state-dict serialization. |
| **Random Seed** | $42$ | Strict deterministic seed for PyTorch, NumPy, and Python RNG. |
| **Execution Device** | `"cpu"` | Guaranteed platform portability without GPU dependency. |

---

## 3. Dynamic Batching & Padding Contract

- **Collation Algorithm**:
  1. For each batch, draw $B=2$ records using deterministic `np.random.RandomState(seed)`.
  2. Encode input and target sequences via `encode_example()`.
  3. Determine maximum sequence length in the current batch: $L_{\text{max}} = \max(|x_1|, |x_2|)$.
  4. Pad inputs dynamically: $x_{\text{padded}} = x + [\text{PAD}] \times (L_{\text{max}} - |x|)$ where $\text{PAD} = 2$.
  5. Pad targets dynamically: $y_{\text{padded}} = y + [-100] \times (L_{\text{max}} - |y|)$.
- **Padding Invariant**: Padding tokens are never included in loss computation (masked with $-100$).
- **No All-Masked Batches**: Collator verifies $|\mathcal{A}_{\text{batch}}| > 0$; fails closed if active tokens $= 0$.

---

## 4. Checkpoint Serialization & Recovery Protocol

### 4.1 Checkpoint Artifacts
Checkpoints are written atomically to `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/checkpoints/`:
- `sft_checkpoint_step_100.pt`
- `sft_checkpoint_step_200.pt`
- `sft_checkpoint_step_300.pt`
- `sft_checkpoint_step_400.pt` (Final step)
- `sft_checkpoint_best.pt` (Global minimum validation loss)

### 4.2 Checkpoint State Dictionary Payload
Each checkpoint `.pt` file must contain:
```python
{
    "sft_step": step,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "config": self.config.__dict__,
    "val_loss": val_loss,
    "val_perplexity": val_perplexity,
    "rng_state": torch.get_rng_state(),
    "numpy_rng_state": rng.get_state(),
    "history": self.history,
    "base_checkpoint_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
```

### 4.3 Atomic Writing Rule
To prevent corrupt checkpoint files upon process interruption, all `.pt` files must be written to a temporary sibling file (`checkpoint.tmp`) and atomically renamed via `os.replace()`.

---

## 5. Weight Parameter Immutability & Delta Tracking

To distinguish true checkpoint file integrity from in-memory parameter drift:
1. **Pre-Training Base Checkpoint Verification**:
   The input checkpoint file `training_checkpoint_step_10000.pt` must match SHA-256 `96357705...`. It is loaded strictly read-only and **never overwritten**.
2. **Post-Training Parameter Delta Verification**:
   After training, the parameter delta tensor norm must be computed and logged:
   $$\Delta_{\mathbf{W}} = \sqrt{\sum_{\theta \in \mathbf{\Theta}} \|\theta_{\text{sft}} - \theta_{\text{base}}\|_2^2}$$
   - *Expected Range*: $0.05 < \Delta_{\mathbf{W}} < 2.50$.
   - If $\Delta_{\mathbf{W}} == 0.0$: Optimizer failed to update parameters.
   - If $\Delta_{\mathbf{W}} > 10.0$: Divergent fine-tuning; catastrophic parameter drift.
