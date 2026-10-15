# Parameter Count Discrepancy & Weight Tying Audit Report
**Location:** `experiments/batch381_410_integrity/reports/PARAMETER_COUNT_AUDIT.md`  
**Milestone:** Batches 381–410  
**Scope:** Reconciling reported counts of 3,443,136 vs 4,229,568 parameters in ChakrMicro v0.1.

---

## 1. Executive Summary: The Root Cause

The reported discrepancy between **3,443,136** parameters (historical documentation) and **4,229,568** parameters (recent inspection) is **NOT** a change in architecture, config, layer count, or weights.

It is a mathematical artifact of **how state_dict serialization handles tied embeddings**:

1. **In-Memory Architecture (`ChakrMicro`):**
   - Embedding layer: `TokenEmbedding` with shape `[4096, 192]` = 786,432 parameters.
   - Output projection head: `LMHead` shares the exact pointer to `self.embedding.weight` (`tie_embeddings=True`).
   - PyTorch `set(model.parameters())` or `list(model.parameters())`: LM head does not introduce a duplicate parameter in the module tree.
   - Total Unique In-Memory Parameters = **3,443,136**.
   - Exactly matches `ChakrMicro.count_parameters()["total_parameters"] = 3,443,136`.

2. **Serialized PyTorch Checkpoint (`model_state_dict`):**
   - Standard PyTorch `model.state_dict()` iterates over module names.
   - Both `'embedding.weight'` and `'lm_head.weight'` are saved as key-value pairs in the checkpoint dictionary.
   - When naïve inspection runs `sum(t.numel() for t in ckpt['model_state_dict'].values())`, it sums both entries:
     $$\text{Total in state\_dict} = 3,443,136 + 786,432 = 4,229,568$$
   - The difference is exactly $4096 \times 192 = 786,432$, representing the duplicated shared weight tensor in the serialized dictionary.

---

## 2. Quantitative Mathematical Breakdown

| Component | Tensor Name / Submodule | Tensor Shape | Unique Elements | Serialized Elements in `state_dict` |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embedding** | `embedding.weight` | `[4096, 192]` | 786,432 | 786,432 |
| **Attention Projections (x6)** | `layers.*.attn.{q,k,v,out}_proj.weight` | 24 tensors of `[192, 192]` | 884,736 | 884,736 |
| **FFN Projections (x6)** | `layers.*.ffn.{gate,up,down}_proj.weight`| 18 tensors | 1,769,472 | 1,769,472 |
| **Layer Normalizations** | `layers.*.norm_{1,2}.weight` + `final_norm.weight` | 13 tensors of `[192]` | 2,496 | 2,496 |
| **LM Head (Tied)** | `lm_head.weight` (shares `embedding.weight`) | `[4096, 192]` | **0** (Tied) | **786,432** (Serialized) |
| **TOTAL** | | | **3,443,136** | **4,229,568** |

---

## 3. Verifiable Reproducibility Script

```python
import torch
from chakrview.brain.model import ChakrMicro

ck = torch.load('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', map_location='cpu', weights_only=False)
sd = ck['model_state_dict']

# Naive sum of state_dict entries:
assert sum(t.numel() for t in sd.values()) == 4229568

# Model instantiation with tied embeddings:
model = ChakrMicro(ck['config'])
model.load_state_dict(sd)

# Parameter iterator respecting weight tying:
assert sum(p.numel() for p in set(model.parameters())) == 3443136
assert model.count_parameters()['total_parameters'] == 3443136
```

---

## 4. Policy & Engineering Directive

- **True Parameter Count:** The authoritative parameter count of ChakrMicro v0.1 remains **3,443,136** unique trainable parameters.
- **Reporting Convention:** All subsequent audits must explicitly distinguish:
  - *Active Trainable Model Parameters:* 3,443,136
  - *Checkpoint Serialized Tensor Elements:* 4,229,568 (including duplicate tied LMHead key)
