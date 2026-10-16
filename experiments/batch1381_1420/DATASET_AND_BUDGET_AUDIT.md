# Dataset and Budget Audit

## Batches 1381–1420: Training Data Volume, Epoch Math and Token Exposure Budget

### 1. Verification of Dataset Splits and Tokens

We verified the tokenized document counts using the frozen vocab-4096 tokenizer:

| Split | Document Count | Total Tokens | Target Tokens ($T - 1$) | Average Tokens / Doc |
|---|---|---|---|---|
| **Full Corpus** | 52 documents | 6,695 tokens | 6,643 tokens | 128.8 tokens |
| **Training Split (Batch 1341–1380)** | 25 documents | 3,212 tokens | 3,187 tokens | 128.5 tokens |
| **Held-Out Test Split (Batch 1341–1380)** | 27 documents | 3,483 tokens | 3,456 tokens | 129.0 tokens |

---

### 2. Validation / Training Partition for Overfitting Protection

To strictly avoid data snooping on the 27 held-out test documents, the 25 training documents are partitioned into:

- **Adaptation Train Split:** 20 documents (2,570 tokens; 2,550 targets)
  - `billing_enterprise_discounts.md`
  - `billing_licensing_compliance.md`
  - `billing_metered_usage.md`
  - `billing_payment_terms.md`
  - `compute_accelerator_cuda.md`
  - `compute_accelerator_metal.md`
  - `compute_arch_core.md`
  - `compute_arch_v07_draft.md`
  - `network_consensus_raft.md`
  - `network_dns_resolution.md`
  - `network_encryption_transit.md`
  - `security_at_rest_encryption.md`
  - `security_audit_logging.md`
  - `security_auth_tokens.md`
  - `sop_cluster_drain.md`
  - `sop_disaster_recovery.md`
  - `sop_document_reindex.md`
  - `storage_backup_policy.md`
  - `storage_blob_store.md`
  - `storage_bm25_index.md`

- **Development Validation Split (for early stopping):** 5 documents (642 tokens; 637 targets)
  - `billing_refund_policy.md`
  - `compute_cpu_simd.md`
  - `network_firewall_rules.md`
  - `security_incident_response.md`
  - `storage_caching_layer.md`

- **Held-Out Test Split (Strictly Unseen):** 27 documents (3,456 targets).

---

### 3. Training Budget Calculations & Risk of Over-Fitting

Let us calculate the token exposure dynamics for a small model (3.44M parameters) trained on 2,550 tokens:

- **Tokens per pass (1 epoch):** 2,550 target tokens.
- **Batch Size:** 4 documents per step (~510 tokens/step).
- **Steps per epoch:** $\approx 5$ optimizer steps.

#### Exposure under various step budgets:
- At **50 steps:** $50 \times 510 = 25,500$ tokens seen $\rightarrow$ **10 passes (epochs)** over the data.
- At **100 steps:** $100 \times 510 = 51,000$ tokens seen $\rightarrow$ **20 passes (epochs)** over the data.
- At **500 steps (original proposal):** $500 \times 510 = 255,000$ tokens seen $\rightarrow$ **100 passes (epochs)** over the data!

#### Overfitting Risk Analysis:
Exposing a 3.44M parameter model to the exact same 20 documents for **100 passes** is guaranteed to produce extreme memorization, where training loss collapses toward zero while validation loss degrades steeply.

**Budget Modification:**
- **Maximum Step Ceiling:** 80 steps (approx. 16 epochs).
- **Validation Frequency:** Every 5 steps.
- **Early Stopping Patience:** 3 evaluations (15 steps) without validation loss improvement.
- **Learning Rate:** Low $\eta = 5\times 10^{-5}$ with warmup over 10 steps and cosine decay to $5\times 10^{-6}$.
- **Weight Decay:** 0.05 to regularize weight drift.
- **Estimated Execution Time:** < 15 seconds on CPU.
