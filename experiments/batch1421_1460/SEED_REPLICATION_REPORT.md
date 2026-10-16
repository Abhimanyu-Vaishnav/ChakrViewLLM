# Seed Replication and Stability Report

## Batches 1421–1460: Multi-Seed Replication of Domain Adaptation on Independent Corpus

### 1. Multi-Seed Replication Protocol

To verify that the domain-adaptation performance gains are not an artifact of random document batch order (Seed 42), the exact 80-step adaptation protocol was independently re-executed using two additional seeds: **Seed 101** and **Seed 202**.

All runs:
- Started from the identical canonical checkpoint `training_checkpoint_step_10000.pt`.
- Used the identical 21 adaptation training documents and 4 validation documents.
- Evaluated on the identical 30 independent held-out documents (7,741 target tokens).

---

### 2. Cross-Seed Metric Comparison on Independent Corpus

| Run / Seed | Checkpoint Path | Training Time | Test Loss (nats) | Test Perplexity ($PPL$) | Top-1 Accuracy | Top-10 Accuracy |
|---|---|---|---|---|---|---|
| **Baseline (No Adaptation)** | `training_checkpoint_step_10000.pt` | — | 5.3852 | **218.15** | 11.02% | 35.95% |
| **Candidate (Seed 42)** | `chakrmicro_domain_adapted_experimental.pt` | 14.20s | 4.8497 | **127.70** | 17.41% | 40.81% |
| **Replication 1 (Seed 101)** | `chakrmicro_domain_adapted_seed_101.pt` | 14.91s | 4.8480 | **127.49** | 17.53% | 40.64% |
| **Replication 2 (Seed 202)** | `chakrmicro_domain_adapted_seed_202.pt` | 14.33s | 4.8513 | **127.91** | 17.59% | 40.59% |

---

### 3. Statistical Repeatability Assessment

- **Perplexity Mean across 3 Seeds:** $127.70 \pm 0.21$
- **Loss Mean across 3 Seeds:** $4.8497 \pm 0.0016$ nats
- **Top-1 Accuracy Mean across 3 Seeds:** $17.51\% \pm 0.09\%$
- **Top-10 Accuracy Mean across 3 Seeds:** $40.68\% \pm 0.12\%$

#### Conclusion:
Variance across distinct batch-ordering seeds is **negligible ($\sigma_{PPL} = 0.21$, $< 0.2\%$)**. This conclusively proves that:
1. The optimization trajectory under the conservative learning rate ($5\times 10^{-5}$) with cosine decay is **extremely stable**.
2. The observed generalization improvement on independent held-out technical text is completely reproducible and robust to batch sampling noise.
