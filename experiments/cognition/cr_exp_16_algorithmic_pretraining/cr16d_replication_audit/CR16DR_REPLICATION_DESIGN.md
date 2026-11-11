# CR-16DR Replication Design & Methodology

### 1. Replication Mission Design
CR-EXP-16D-R was executed to test the reproducibility and stability of CR-16D's findings through:
1. Complete independent forensic audit of historical artifacts.
2. Targeted Part B diagnostics (counterfactual pairs, position permutations, distractors, logit analysis).
3. Independent replication training across **3 new seeds (202, 303, 404)** across all 4 experimental arms.

Combined with historical seeds 42 and 101, this provides **5 independent random seeds per arm ($N=20$ total training runs)**.

---

### 2. Experimental Arms & Hyperparameters

| Arm | Description | Copy Samples | Retr Samples | Total Samples | Steps | Batch Size | LR Schedule |
|---|---|---|---|---|---|---|---|
| **Arm A** | 0% Replay Control | 0 | 4,000 | 4,000 | 400 | 16 | $2\times 10^{-4}$ (cos decay) |
| **Arm B** | 25% Copy Replay | 1,000 | 3,000 | 4,000 | 400 | 16 | $2\times 10^{-4}$ (cos decay) |
| **Arm C** | 50% Copy Replay | 2,000 | 2,000 | 4,000 | 400 | 16 | $2\times 10^{-4}$ (cos decay) |
| **Arm D** | 75% Copy Replay | 3,000 | 1,000 | 4,000 | 400 | 16 | $2\times 10^{-4}$ (cos decay) |

- **Parent Checkpoint:** CR-16A Arm D generalization checkpoint (`9d609a77...`) across all 12 replication runs.
- **Architecture:** Strictly frozen canonical ChakrMicro v0.1 ($d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$).
- **Hardware:** NVIDIA RTX A1000 Laptop GPU (6.0 GB VRAM).
- **Execution:** GPU-accelerated local training via `.venv_gpu`.

