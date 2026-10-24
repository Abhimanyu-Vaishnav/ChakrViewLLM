# ChakrView Exp18 — Training Protocol Specification

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/TRAINING_PROTOCOL.md`  
**Engine**: `chakrview/training/sft_engine.py` (Local Stage D SFT Engine)  
**Execution Device**: Local CPU (`device = "cpu"`)  
**Status**: SPECIFICATION ONLY — NO OPTIMIZER STEPS AUTHORIZED IN THIS PHASE  

---

## 1. Mathematical Objective & Loss Masking

Exp18 trains the `ChakrMicro` causal autoregressive language model using **Prompt-Masked Cross-Entropy Loss**.

### 1.1 Target Masking Alignment
For an input sequence $\mathbf{x} = [\text{BOS}, p_1, \dots, p_L, r_1, \dots, r_M]$:
$$\mathbf{y} = [\underbrace{-100, -100, \dots, -100}_{L \text{ prompt tokens}}, r_1, r_2, \dots, r_M, \text{EOS}]$$

- Token positions corresponding to the prompt $\mathbf{p}$ are masked with index `-100`.
- The loss is computed strictly across the response tokens $r_1, \dots, r_M$ and the final `EOS` token (token ID `1`).
- Gradient flow is strictly confined to response synthesis and turn-boundary termination.

### 1.2 Loss Formulation
$$\mathcal{L}(\theta) = - \frac{1}{\sum_{i=1}^B |\mathcal{A}_i|} \sum_{i=1}^B \sum_{t \in \mathcal{A}_i} \log P_\theta(y_{i,t} \mid x_{i,\le t})$$
where $\mathcal{A}_i = \{t \mid y_{i,t} \ne -100\}$.

---

## 2. Hyperparameter Schedule & Optimization Plan

| Hyperparameter | Value | Rationale & Invariant |
| :--- | :--- | :--- |
| **Base Model Checkpoint** | `step_10000.pt` | Canonical checkpoint (SHA-256: `96357705...`) |
| **Optimizer** | AdamW | Decoupled weight decay ($0.01$) |
| **Peak Learning Rate** | $5.0 \times 10^{-5}$ | Conservative peak learning rate to preserve pretraining representations |
| **Minimum Learning Rate** | $5.0 \times 10^{-6}$ | 10% of peak learning rate |
| **Learning Rate Schedule** | Cosine Annealing | Decays across 1,200 optimizer steps with 50-step linear warmup |
| **Batch Size ($B$)** | 2 sequences | Optimal cache locality and stable gradient estimates on CPU |
| **Gradient Accumulation** | 2 steps | Effective batch size = 4 sequences |
| **Total Optimizer Steps** | 1,200 steps | 2.0 epochs across 1,200 training examples ($1200 / (2 \times 2) \times 2 = 600$ updates) |
| **Gradient Clipping** | 1.0 (`max_norm`) | Prevents gradient explosion on complex JSON syntax tokens |
| **Evaluation Interval** | Every 50 steps | Validates against all 160 validation examples |
| **Checkpoint Strategy** | Top-3 val loss + step 600, 1200 | Saved atomically as `exp18_sft_step_*.pt` |
| **Random Seed** | 42 | Deterministic initialization across Python, NumPy, and PyTorch |

---

## 3. Resource Requirements & Failure Safeguards

### 3.1 Hardware Budgeting
- **Device**: CPU only (`torch.device("cpu")`). Multi-threading capped at available physical cores (`torch.set_num_threads(...)`).
- **Memory**: Peak RSS memory $\le 1.5 \text{ GB}$.
- **Storage**: Checkpoints stored under `exp18_scaled_instruction_format_v1/checkpoints/` ($\le 100 \text{ MB}$ total).

### 3.2 Automated Training Halts
Training automatically aborts under any of the following conditions:
1. `loss != loss` (NaN detected in loss or gradient norm).
2. Validation loss increases continuously for 5 consecutive checkpoints (overfitting safeguard).
3. Any attempt to write to canonical checkpoint or source files outside the experiment folder.
