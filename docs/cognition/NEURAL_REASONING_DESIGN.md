# NEURAL REASONING DESIGN: ARCHITECTURAL MECHANISMS (CR-001)

This document systematically evaluates 8 candidate neural mechanisms for instantiating first-class reasoning within ChakrView.

---

## 1. Mechanism Evaluation Matrix

| Proposal | Expected Benefits | Parameter / Compute Cost | CPU & RAM Impact | Compatibility with ChakrMicro | Risks & Failure Modes | Required Pre-Experiment |
|---|---|---|---|---|---|---|
| **1. Standard Backbone + Dense Reasoning Curriculum** | Maximizes inductive learning without architecture changes. | 0 added parameters (3.44M). Zero added compute per token. | Baseline (<40 MB RAM, >1,300 tok/s). | 100% bit-exact baseline compatibility. | Limited capacity in 6 layers may saturate attention heads. | CR-EXP-01 (GSM8K & synthetic induction curriculum). |
| **2. Structured Intermediate Scratchpad Tokens** | Forces autoregressive attention to compute sequential state transitions explicitly. | 0 added parameters (uses special vocab delimiter tokens). | Sequence length increases by $2\times\text{--}3\times$, inference latency scales linearly. | 100% compatible with existing vocab & RoPE. | Slower generation; vulnerability to intermediate token loops. | CR-EXP-02 (Chain-of-logic delimiter tokens). |
| **3. Auxiliary Consistency / Validity Heads** | Predicts binary validity $V \in \{0, 1\}$ or entailment directly from hidden states $h_L$. | ~30,000 parameters (192 $\times$ 2 linear layer). | Negligible (<0.1 MB RAM, <1% compute overhead). | Requires isolated variant (ChakrMicro-Aux); cannot alter frozen baseline. | Gradient interference between LM loss and auxiliary classification loss. | CR-EXP-03 (Entailment prediction loss ablation). |
| **4. Multi-Task Reasoning Pre-Training** | Prevents conversational degradation by co-training language, logic, and code. | 0 added parameters. | Same training throughput; requires balanced sharded batches. | 100% compatible. | Complex loss balancing; negative transfer between domains if unbalanced. | CR-EXP-04 (Tri-domain interleaved pretraining wave). |
| **5. Internal Working-Memory State (Latent Slot Memory)** | Dedicated learned persistent memory slots attendable across turn boundaries. | ~100k parameters (memory slot matrices + cross-attention projection). | Increases KV-cache footprint by ~5–10 MB. | Modifies TransformerBlock attention path (requires ChakrMicro-Mem branch). | Memory drift, over-attending to stale historical slots. | CR-EXP-05 (Slot-memory prototype on synthetic copy tasks). |
| **6. Latent Recurrent Refinement (Weight-Tied Loops)** | Passes hidden state $h_L$ back through layers $3\text{--}6$ for $K$ iterations before output projection. | 0 added parameters (reuses existing layer weights). | $K\times$ inference latency ($K=2$ or $3$ drops throughput to ~400 tok/s). | Architecture variant (ChakrMicro-Recur). Baseline weights compatible. | Vanishing/exploding activations across loops without custom layernorms. | CR-EXP-06 (Recurrent loop depth ablation on arithmetic). |
| **7. Explicit Uncertainty Head (Logit Dispersion / Entropy)** | Directly predicts epistemic uncertainty $\sigma$ to trigger abstention internally. | ~1,000 parameters (single linear scalar probe). | Zero measurable CPU overhead. | Isolated experimental head. | Head over-predicts high uncertainty on complex phrasing. | CR-EXP-07 (Calibrated refusal on unanswerable queries). |
| **8. Self-Correction Verification Loop (Two-Pass Generation)** | Model generates candidate answer $\to$ evaluates $\to$ amends in single forward graph. | 0 added parameters. | $2\times$ inference cost per query. | Runtime orchestration or single packed sequence. | Error amplification (model reinforces its own mistake). | CR-EXP-08 (Critic-and-revision supervised fine-tuning). |

---

## 2. In-Depth Mechanism Analysis

### Mechanism 1 & 2: Dense Curriculum + Scratchpad (Immediate Recommendation)
- **Principle**: Mechanistic interpretability demonstrates that small 6-layer transformers possess the mathematical capacity to form 2-layer induction circuits and execute 2-digit arithmetic, provided the training data exposes the sequential state transitions explicitly.
- **Why it fits ChakrView**: Preserves the frozen 3.44M ChakrMicro architecture with 100% fidelity. It requires zero architectural modifications and zero host RAM increase.

### Mechanism 3: Auxiliary Validity Projection (Phase 2 Recommendation)
- Attaching a linear classification head $W_{valid} \in \mathbb{R}^{d_{model} \times 2}$ to the final representation $h_L$ forces the transformer's hidden representations to separate valid logical derivations from invalid ones in geometric space.
- Training objective: $\mathcal{L}_{total} = \mathcal{L}_{causal} + \lambda \mathcal{L}_{valid}$.

### Mechanism 6: Weight-Tied Latent Recurrence (Future Frontier)
- Standard transformers have fixed computational depth $L$. Hard reasoning problems require variable computational depth.
- By allowing layers $3\dots 6$ to repeat their transformation on difficult tokens without adding new weights, the model can simulate a 12-layer or 18-layer transformer while maintaining a 3.44M parameter footprint on CPU.

