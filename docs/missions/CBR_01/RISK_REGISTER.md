# CBR-01 — RISK REGISTER & MITIGATION MATRIX

---

| Risk ID | Description | Severity | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **R1** | **Dataset Semantic Collapse** | HIGH | MEDIUM | Model memorizes static strings instead of learning syntax and variable binding. | Enforce $\ge 90\%$ unique responses in training datasets; reject mixtures with duplicated responses across prompt variants. |
| **R2** | **Attractor Loop Divergence** | HIGH | MEDIUM | Model enters repetitive phrase cycles during greedy generation. | Apply sliding-window n-gram suppression ($N=3$) and SFT training with EOS termination supervision. |
| **R3** | **Evaluator False Positives** | MEDIUM | LOW | Loose substring matching gives false impression of intelligence. | Use exact-word boundary matching, isolated AST execution, and manual audit checks. |
| **R4** | **Hardware Resource Exhaustion** | MEDIUM | LOW | Training exhausts RAM or runs out of disk storage. | Hard limit: batch size 2, gradient accumulation 2, disk usage budget $\le 2\text{ GB}$. |
| **R5** | **Canonical Artifact Mutation** | CRITICAL | LOW | Inadvertent modification of baseline checkpoints or runtime files. | Pre- and post-flight cryptographic SHA-256 verification of all protected files. |
| **R6** | **Test Set Contamination** | HIGH | LOW | Training on held-out evaluation benchmarks. | Strict directory isolation and SHA-256 manifest checks separating training from evaluation benchmarks. |
