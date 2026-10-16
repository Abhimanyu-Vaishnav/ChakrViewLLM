# Parameter Count Reconciliation

## Batches 1381–1420: Rigorous Verification of ChakrMicro Neural Core Weights and Dimensions

### 1. The Discrepancy

Earlier reports in Batches 1261–1340 occasionally referenced a *"10.8M-parameter core"*, while other reports cited *4,229,568 state-dict parameters* and *3,443,136 unique parameters*.

This audit examines the exact code, tensor shapes, weight-sharing mechanics, and documentation origin to resolve this discrepancy conclusively.

---

### 2. Root Cause: Synthetic Documentation Text vs. Concrete Model Implementation

1. **The Origin of "10.8 Million":**
   In Batch 1141–1180, synthetic technical documentation was generated as test corpora for the BM25 local knowledge assistant ([compute_arch_core.md](file:///D:/Project/ChakrView/experiments/batch1141_1180/data/corpus/compute_arch_core.md)). That document stated:
   ```markdown
   # ChakrMicro V0.6 Core Topology
   - Hidden dimension: 384 units.
   - Multi-head attention count: 6 query heads and 2 key-value heads.
   - Intermediate feed-forward layer size: 1536 dimensions.
   - Transformer decoder layers: 6 homogeneous blocks.
   - Weight parameter volume: 10.8 million trainable float parameters.
   ```
   A theoretical model matching these fictional specifications ($d=384, L=6, \text{FFN}=1536, V=4096$) would have roughly 10.8 million parameters. However, **this model was a fictional hardware specification authored inside a test markdown file**, which previous agentic milestone summaries mistakenly cited as the actual model parameter count!

2. **The Concrete Checkpoint Implementation:**
   The actual canonical checkpoint evaluated across Batches 16–21 and Batches 1021–1380 is:
   `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).

   The checkpoint configuration is frozen in `ModelConfig`:
   - `vocab_size`: 4096
   - `d_model`: 192
   - `n_layers`: 6
   - `n_heads`: 6
   - `hidden_dim`: 512 (SwiGLU)
   - `max_seq_len`: 512
   - `tie_embeddings`: True

---

### 3. Exact Mathematical Parameter Breakdown

Let us compute the exact parameter breakdown of the instantiated `ChakrMicro` model:

| Layer / Sub-block | Components & Tensor Shapes | Number of Elements |
|---|---|---|
| **Embedding Layer** | `embedding.weight`: $[4096, 192]$ | $786,432$ |
| **Final RMSNorm** | `final_norm.weight`: $[192]$ | $192$ |
| **Output Head (LM Head)** | `lm_head.weight`: $[4096, 192]$ | $786,432$ (Tied to embedding) |
| **6 $\times$ Transformer Blocks:** | Per layer: | |
| - *Pre-Norm 1* | `norm_1.weight`: $[192]$ | $192$ |
| - *Self-Attention Query* | `q_proj.weight`: $[192, 192]$ | $36,864$ |
| - *Self-Attention Key* | `k_proj.weight`: $[192, 192]$ | $36,864$ |
| - *Self-Attention Value* | `v_proj.weight`: $[192, 192]$ | $36,864$ |
| - *Self-Attention Output* | `out_proj.weight`: $[192, 192]$ | $36,864$ |
| - *Pre-Norm 2* | `norm_2.weight`: $[192]$ | $192$ |
| - *SwiGLU Gate Proj* | `gate_proj.weight`: $[512, 192]$ | $98,304$ |
| - *SwiGLU Up Proj* | `up_proj.weight`: $[512, 192]$ | $98,304$ |
| - *SwiGLU Down Proj* | `down_proj.weight`: $[192, 512]$ | $98,304$ |
| **Total per Transformer Block** | | $442,752$ elements |
| **Total for 6 Blocks** | $6 \times 442,752$ | **$2,656,512$ elements** |

#### Summary of Reconciled Counts:
1. **Total parameter elements represented in state dictionary:**
   $$2,656,512 + 192 + 786,432 (\text{embedding}) + 786,432 (\text{head}) = \mathbf{4,229,568}$$
2. **Unique parameters (accounting for tied embeddings):**
   $$4,229,568 - 786,432 = \mathbf{3,443,136}$$
3. **Trainable parameters:** $\mathbf{3,443,136}$ (all unique weights are `requires_grad=True`).
4. **Frozen parameters:** $\mathbf{0}$.
5. **In-Memory Size (float32):** $3,443,136 \times 4 \text{ bytes} \approx \mathbf{13.77 \text{ MB}}$.
6. **Checkpoint File on Disk:** $17.1 \text{ MB}$ (includes optimizer state, seed, metadata).

---

### 4. Conclusion and Invariant Policy

- The true, physical ChakrMicro neural core has **3,443,136 unique parameters** (4.23M serialized elements).
- Any future milestone report must cite **3.44M unique parameters**, and explicitly note that the "10.8M" figure was a synthetic document string from `compute_arch_core.md`.
