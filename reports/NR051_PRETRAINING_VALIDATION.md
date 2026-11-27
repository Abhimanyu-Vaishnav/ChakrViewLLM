# NR-051: Pretraining Validation & Objective Audit

**Campaign:** CHAKRVIEW NR-051 — True Language Pretraining & Controlled Scaling  
**Date:** November 27, 2026  
**Artifact Directory:** `experiments/pretraining_nr051/`  
**Test Script:** [test_pretraining_objective.py](file:///d:/Project/ChakrView/experiments/pretraining_nr051/test_pretraining_objective.py)  

---

## 1. Objective Formulation & Verification

Before running pretraining on either architecture, the causal next-token prediction objective and tensor dynamics were formally verified to ensure no regression or data corruption occurred.

### 1.1 Mathematical Formulation

For input token sequence $X = (x_1, x_2, \dots, x_T)$, the causal autoregressive target sequence is $Y = (x_2, x_3, \dots, x_T, \text{pad})$.
The loss function is formulated as:
$$\mathcal{L}_{PT}(\theta) = -\frac{1}{\sum_{t=1}^{T-1} \mathbb{I}(y_t \ne -100)} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \dots, x_t)$$

where $\mathbb{I}(\cdot)$ masks out padding positions.

### 1.2 Verification Test Matrix

Both `ChakrGenesis17M` and `ChakrGenesis38M` were instantiated and tested against `test_pretraining_objective.py`. The following criteria were empirically validated:

1. **Target Shift Correctness:** Verified that output logit at index $t$ corresponds strictly to predicting target token $t+1$. Off-by-one shifts evaluated to $0$ error.
2. **Padding Invariance:** Batches containing padded suffixes verified that padding tokens assigned `ignore_index = -100` produce zero gradients and do not alter the loss denominator.
3. **Causal Attention Masking:** Verified that upper triangular entries in the attention score matrix are $-\infty$ (`torch.finfo(torch.float32).min`), guaranteeing zero future-token information leakage.
4. **Gradient Health:** Backpropagation checked across all layers. All gradient norms remained finite ($0 < \|\nabla_\theta\| < \infty$, with zero `NaN` or `Inf` occurrences).

---

## 2. Pretraining Execution Dynamics

Both models were trained using AdamW with cosine decay, linear warmup (100 steps), weight decay $0.01$, and mixed precision (FP16 autocast) on the NVIDIA RTX A1000 Laptop GPU.

```
+---------------------------------------------------------------------------------------+
| Model                  | Parameters | Best Val Loss | Val PPL | Val Acc | Peak VRAM   |
+------------------------+------------+---------------+---------+---------+-------------+
| ChakrGenesis-17M Base  | 17,063,808 | 3.0516        | 21.15   | 53.21%  | 1,525.0 MB  |
| ChakrGenesis-38M Base  | 37,499,392 | 2.8502        | 17.29   | 55.37%  | 2,103.1 MB  |
+------------------------+------------+---------------+---------+---------+-------------+
```

### 2.1 Learning Trajectory Highlights

- **ChakrGenesis-17M Base:**
  - Started at Step 100 with Val Loss $5.412$, Val Acc $14.2\%$.
  - Reached Val Loss $3.0516$ at Step 529 (PPL $21.15$), with final validation token accuracy of $53.21\%$.
  - Checkpoint SHA-256: `8d871ca0aff6e23398bdb17f86521eb26dfb4b778d227035d77e2fe44803c25f`.

- **ChakrGenesis-38M Base:**
  - Started at Step 100 with Val Loss $5.276$, Val Acc $16.9\%$.
  - Reached Val Loss $2.8502$ at Step 529 (PPL $17.29$), with final validation token accuracy of $55.37\%$.
  - Checkpoint SHA-256: `22306c9a479c80cd9ce3e937e3145fb1520241d96c3b17b3ff8aed2af4d7f304` (saved to `experiments/pretraining_nr051/chakr_38m_pretrained_base.pt`).

---

## 3. Findings

1. **Pretraining Objective Validity:** Causal language pretraining successfully reduced validation perplexity from $>195$ down to $17.29$ on continuous text.
2. **Predictable Scaling:** Under strictly matched token budgets, the 38M parameter architecture achieved a $-0.2014$ lower validation cross-entropy loss and $+2.16\%$ higher next-token accuracy than 17M.
