# CV-GENESIS-NR-010 17M Training Configuration

## 1. Model Configuration (`Genesis17MConfig`)

```python
class Genesis17MConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 10
    n_q_heads: int = 12
    n_kv_heads: int = 3
    hidden_dim: int = 1024
    max_seq_len: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
```

---

## 2. Training Hyperparameters

- **Hardware:** NVIDIA RTX A1000 Laptop GPU (6.0 GB VRAM)
- **Batch Size:** 16 (effective batch = 4,096 tokens per step)
- **Sequence Length:** 256 tokens
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight decay}=0.01$)
- **Learning Rate:**
  - Pretraining: $4.0 \times 10^{-4}$ with 5% linear warmup and cosine decay to $4.0 \times 10^{-5}$
  - SFT: $6.0 \times 10^{-5}$ with 10% linear warmup and cosine decay to $6.0 \times 10^{-6}$
- **Precision:** PyTorch native automatic mixed precision (`torch.amp.autocast('cuda')`)
- **Gradient Clipping:** Max norm 1.0
