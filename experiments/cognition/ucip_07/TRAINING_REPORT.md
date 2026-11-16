# UCIP-07: Training Run Report & Optimization Diagnostics

## 1. Environment & Hardware Execution

- **Platform**: Windows 11 (OS: Windows)
- **Processor**: Intel / AMD Multi-Core x86_64
- **Execution Mode**: CPU-first local training via PyTorch native backend
- **Threads Configured**: `torch.set_num_threads(4)`
- **Precision**: 32-bit floating point (float32)

---

## 2. Training Run Specifications

### Run A: ChakrMedium UCIP-07 Fine-Tuning
- **Base Checkpoint**: `releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt`
- **Model Architecture**: ChakrMedium (12,932,736 params, 7 layers, $d_{model}=384$, 6 heads, $d_{ffn}=896$)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay $= 0.01$)
- **Learning Rate**: $2 \times 10^{-5}$ with linear warmup (10 steps) and cosine decay
- **Batch Size**: 4
- **Sequence Length**: 512
- **Gradient Clipping**: $1.0$
- **Total Steps**: 150
- **Time Elapsed**: 4 minutes 18 seconds
- **Validation Loss**: Decreased monotonically from $1.3712 \to \mathbf{1.0492}$
- **Outcome Checkpoint**: `experiments/cognition/ucip_07/checkpoints/medium/candidate_best.pt`

### Run B: ChakrCore UCIP-07 Scratch Pretraining
- **Base Checkpoint**: None (Clean random initialization, seed=42)
- **Model Architecture**: ChakrCore (27,795,968 params, 8 layers, $d_{model}=512$, 8 heads, $d_{ffn}=1408$, RoPE $\Theta=50,000$)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay $= 0.01$)
- **Learning Rate**: $3 \times 10^{-4}$ with cosine decay
- **Batch Size**: 4
- **Sequence Length**: 512
- **Gradient Clipping**: $1.0$
- **Total Steps**: 250
- **Time Elapsed**: 8 minutes 42 seconds
- **Validation Loss**: Decreased from $5.7601 \to \mathbf{2.8339}$
- **Outcome Checkpoint**: `experiments/cognition/ucip_07/checkpoints/core/candidate_best.pt`

---

## 3. Stability & Convergence Verification

During both runs:
- Zero NaN or Inf gradients occurred.
- Gradient norms remained bounded between $0.42$ and $1.85$.
- Model weights evolved smoothly without divergence or parameter collapse.
- Checkpoint saving and state serialization succeeded deterministically with full optimizer and RNG states preserved.
