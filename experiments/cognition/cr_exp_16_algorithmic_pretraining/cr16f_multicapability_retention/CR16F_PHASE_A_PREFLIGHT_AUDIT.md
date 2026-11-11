# CR-16F Phase A: Pre-Experiment Forensic Verification Report

### 1. Verification of Canonical Safety Invariants

Every non-negotiable canonical artifact was verified bit-for-bit:

| Target / Artifact | Canonical Path | Expected SHA-256 Digest | Actual SHA-256 Digest | Status |
|---|---|---|---|:---:|
| **Canonical Baseline Architecture Init** | `chakrview/brain/model.py` (seed=42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **EXACT MATCH** |
| **Canonical Stage B Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **EXACT MATCH** |
| **CR-16A Arm D Parent Checkpoint** | `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | **EXACT MATCH** |
| **Frozen CR-16 Copy Benchmark** | `experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json` | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | **EXACT MATCH** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNMODIFIED** |

---

### 2. Forensic Audit of CR-16E-R2 Lineage & Training Setup

- **Parent Checkpoint:** CR-16A Arm D (`9d609a77...`), $3,443,136$ parameters ($d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$).
- **Optimizer Configuration:** AdamW, learning rate $2 \times 10^{-4}$, weight decay $0.01$, betas $(0.9, 0.95)$.
- **Batch Size & Steps:** Batch size $16$, gradient clipping norm $1.0$, trained for $400$ steps ($250$ batches per epoch).
- **CR-16E-R2 Checkpoints Verified:**
  - `arm_g1_enabled_seed2026/arm_g1_enabled_best.pt`: `0b6693915f1a532d85d5c11c136e6c13642c5ae0b2772793a42c81013a974b53`
  - `arm_g1_enabled_seed3030/arm_g1_enabled_best.pt`: `375ea414e7cc2ac90a7be0feaf661b26bc8a0b71987ef8eb3e05d272692494a1`
  - `arm_g1_enabled_seed4040/arm_g1_enabled_best.pt`: `55cf88c1fc16a98019af2b5645c46c69c63db16987ab02b503be2eb211ef7835`
  - `arm_g2_masked_seed2026/arm_g2_masked_best.pt`: `08f82d2a5be713012cad1f891b8790aadfb3e7f3fd2a6a4541bf9f5daf36cfd9`
  - `arm_g2_masked_seed3030/arm_g2_masked_best.pt`: `0f1e555b210085004aba6703926d981ee48fab380bd09d731901e96bba7a519e`
  - `arm_g2_masked_seed4040/arm_g2_masked_best.pt`: `c3712ce3c833ef762f97d17c72268bc66b554fbf9200a6fe854900ef02a767fc`

---

### 3. Training Data Composition & Batch-Level Replay Definition

To ensure reproducible, auditable conditions for CR-16F across Arms A, B, C, D:
- **Batch Size:** Exactly $16$ samples per step.
- **25% Copying Replay Condition (Arms A & C):**
  - Exactly **4 copying examples** and **12 retrieval/abstention examples** per batch.
  - Of the 12 retrieval examples, negative abstention examples are maintained in their exact $12.5\%$ dataset ratio ($1.5$ per batch, or exactly $375$ across $2,500$ retrieval examples).
- **50% Copying Replay Condition (Arms B & D):**
  - Exactly **8 copying examples** and **8 retrieval/abstention examples** per batch.
  - Negative abstention examples are maintained in their exact ratio ($1.0$ per batch).
- **Negative Abstention Loss Multiplier ($\lambda$):**
  - Arms A & B: $\lambda = 1.00$.
  - Arms C & D: $\lambda = 0.25$.
- Deterministic batch streams generated seed-by-seed ensuring identical example sequences.

