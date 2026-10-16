# Kernel Fusion and Graph Compilation

## 1. Optimization Passes
- Fused operations: LayerNorm plus Linear projection kernel fusion.
- Activation fusion: GELU activation folded directly into feed-forward linear layers.
- Graph optimization speedup: 18.5% median latency reduction on CPU.
