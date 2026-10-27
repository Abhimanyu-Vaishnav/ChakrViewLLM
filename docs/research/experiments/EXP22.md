# Research Evidence Dossier: Experiment 22 (Exp22)
## Attention Copy Loss Treatment vs Control

- **Archive Status**: COMMITTED_IN_GIT
- **Artifact Status**: LOCAL_ONLY (Treatment Evaluation Metrics)
- **Primary Evidence Directory**: experiments/overnight_runs/exp22_attention_copy_loss_cpu_300_20261024_212632/

---

### 1. Purpose and Hypothesis
- **Purpose**: Evaluate an auxiliary attention loss targeting attention maps directly towards source copy tokens during training.
- **Treatment vs Control**: Control used standard cross-entropy; Treatment introduced weighted attention regularization on copy spans.

### 2. Empirical Findings
- Treatment showed accelerated attention convergence, but did not translate to superior final accuracy over baseline cross-entropy once learning rates were decayed.
