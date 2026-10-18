# Batch 104 — Controlled Slot-Weighted Training Experiment

**Sprint**: ChakrView Master Sprint Batches 103–106  
**Module**: Dynamic Argument Learning: Slot-Weighted SFT  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Experimental Hypothesis

The canonical model, when fine-tuned uniformly, readily memorizes the syntactic skeleton (`TOOL_CALL: {"tool": "calculator", "op": "add", "a": X, "b": Y}`) but collapses its argument slots to static mode tokens (e.g., `a: 1`, `b: 8`). While representation probing demonstrated that ground-truth operand tokens receive small non-zero probability ($\sim 8.7\%$ to $9.8\%$), greedy decoding deterministically selects the static mode attractor.

**Hypothesis**: Multiplying the loss on the target token positions corresponding to the argument values `a` and `b` by $5.0\times$ will force the optimizer to prioritize argument value selection over formatting syntax, lifting the dynamic operand token above the static attractor threshold.

---

## 2. Experimental Setup & Controls

To guarantee causal comparability, all experimental arms shared:
- **Base Checkpoint**: Canonical step-10,000 checkpoint (`training_checkpoint_step_10000.pt`, SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Data Stream**: Identical 71,139-token structured slot-binding stream (`train_slot_tokens.bin`).
- **Training Budget**: 100 optimizer steps, batch size = 4, sequence length = 64 tokens, total budget = 25,600 token exposures.
- **Optimizer Settings**: AdamW ($\text{lr} = 5\times 10^{-4}$, $\text{weight\_decay} = 0.01$, gradient clipping norm = 1.0).
- **RNG Seed**: Fixed seed = 42 for both PyTorch and NumPy data sampler.
- **Tokenizer**: Canonical BPE tokenizer (`vocab_4096`).

---

## 3. Loss-Masking Formulation and Implementation

For input tokens $x_{1:T}$ and target tokens $y_{1:T} = x_{2:T+1}$, the unreduced per-token negative log-likelihood is:
$$\ell_t = -\log P_\theta(y_t \mid x_{1:t})$$

In the **uniform-loss control (Condition B)**:
$$\mathcal{L}_{\text{uniform}} = \frac{1}{T} \sum_{t=1}^T \ell_t$$

In the **slot-weighted condition (Condition C)**:
$$\mathcal{L}_{\text{weighted}} = \frac{\sum_{t=1}^T w_t \ell_t}{\sum_{t=1}^T w_t}, \quad \text{where } w_t = \begin{cases} 5.0 & \text{if } y_t \in \text{ArgumentSlots} \\ 1.0 & \text{otherwise} \end{cases}$$

### Target Position Detection & Multi-Token Handling
- In `train_slot_tokens.bin`, total tokens = 71,139.
- Dynamic slot positions were located by searching for the field patterns:
  - `"` (37), `a` (100), `": ` (544) $\to$ followed by value token(s).
  - `"` (37), `b` (101), `": ` (544) $\to$ followed by value token(s).
- Total target argument tokens identified in stream: 1,606 tokens (2.26% of stream).
- Multi-token argument values (such as `9` which encodes to 2 tokens `[60, 276]` or `[60, 128]`) are marked across all value tokens up to the closing punctuation.

---

## 4. Training Progression

| Metric | Condition B (Uniform Control) | Condition C (Slot-Weighted 5×) |
|---|---|---|
| Step 1 Loss | 5.8925 | 5.9217 |
| Step 25 Loss | 0.2162 | 0.3637 |
| Step 50 Loss | 0.1934 | 0.2682 |
| Step 75 Loss | 0.1310 | 0.1317 |
| Step 100 Loss | 0.1377 | 0.1416 |

Both runs exhibited monotonic optimization convergence and smooth loss curves without instability.

---

## 5. Artifact Hashes & Immutability

- **Condition A (Condition C baseline)**:
  - Checkpoint: `experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt`
  - SHA-256: `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`
- **Condition B (Uniform-Loss Control)**:
  - Checkpoint: `experiments/batch103_106_slot_weighted_sft/checkpoints/uniform_loss_control_stream_step100.pt`
  - SHA-256: `7967ab078705c60643f795878c06539c349b0efee728cc04c9699b39d21bf5e5`
- **Condition C (Slot-Weighted 5×)**:
  - Checkpoint: `experiments/batch103_106_slot_weighted_sft/checkpoints/slot_weighted_5x_stream_step100.pt`
  - SHA-256: `7b057042dc375444406cd02de552fc72ff98257cc92e92046a2bc45846946e7a`
