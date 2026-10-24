# Exp20 Training Dynamics & Architecture Reconciliation Audit

**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  

---

## 1. Resolution of the Architecture Dimension Discrepancy

In the post-experiment report (`EXP20_COMPARISON.md`), a section described ChakrMicro as:
`"ChakrMicro (3.44M parameters, 6 layers, 384 embedding dim, 6 attention heads)"`

### Audit Investigation
1. `chakrview/brain/config.py`:
   - `d_model: int = 192` (frozen default).
   - `n_layers: int = 6`
   - `n_heads: int = 6`
   - `hidden_dim: int = 512`
   - Total parameter count: **Exactly 3,443,136 parameters**.
2. Saved Checkpoint `sft_best_checkpoint.pt`:
   - `embedding.weight`: `torch.Size([4096, 192])`
   - `layers.0.attn.q_proj.weight`: `torch.Size([192, 192])`
   - `layers.0.ffn.gate_proj.weight`: `torch.Size([512, 192])`

### Finding
The mention of `"384 embedding dim"` in the narrative prose was a **typographical error**. The actual model instantiated, trained, and evaluated in Exp20 was the strict canonical ChakrMicro architecture with `d_model = 192` and exactly **3,443,136 parameters**. No architecture modification occurred.

---

## 2. Training Log and Checkpoint Consistency

- **Configured Steps**: 400 optimizer steps.
- **Completed Steps**: 400 optimizer steps.
- **Logging Intervals**: 20 validation checks (evaluated every 20 steps).
- **Global Validation Loss Minimum**: **1.7447 at Step 240** (Perplexity: 5.72).
- **Final Validation Loss**: **2.3087 at Step 400** (Perplexity: 10.06).
- **Best Checkpoint State Dict**: `sft_best_checkpoint.pt` matches Step 240 weights with zero diffs.
- **Checkpoint Separation**: The canonical Step 10,000 checkpoint was completely untouched; experimental checkpoints exist strictly within `checkpoints/`.
