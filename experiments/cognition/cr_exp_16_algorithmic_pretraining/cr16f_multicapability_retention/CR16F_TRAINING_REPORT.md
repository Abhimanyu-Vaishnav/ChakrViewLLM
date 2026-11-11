# CR-EXP-16F: 12 Factorial Models Training Report

## 1. Executive Summary

CR-EXP-16F investigated whether controlled combinations of sequence-copying replay and abstention loss weighting can prevent catastrophic forgetting while acquiring key-value retrieval and calibrated abstention.

All twelve models in the $2 \times 2$ factorial experimental design across $N=3$ random seeds (2026, 3030, 4040) were trained strictly from the canonical parent checkpoint:
- **Parent Checkpoint:** CR-16A Arm D (`9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5`)
- **Parent Architecture:** ChakrMicro v0.1 frozen (6 layers, 192 dim, 6 heads, 512 mlp, 4096 vocab, 3,443,136 parameters; seed 42 parameter digest: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`)
- **Protected File:** `chakrview/runtime/session.py` (`9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`) strictly untouched.

Training executed to completion across all 12 experimental runs without interruption, gradient explosion, or NaN values.

---

## 2. Experimental Design Matrix & Checkpoint Lineage

The study tests two factors:
1. **Replay Fraction:** 25% (4 copying samples per batch of 16) vs. 50% (8 copying samples per batch of 16).
2. **Negative Abstention Loss Weight ($\lambda$):** 1.00 (Standard unreduced CE) vs. 0.25 (Scaled down negative loss).

| Run / Model Name | Replay % | $\lambda_{\text{neg}}$ | Seed | Best Step | Best Val Loss | Val Token Acc | Checkpoint SHA-256 Digest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `arm_a_replay25_lambda100_seed2026` | 25% | 1.00 | 2026 | 400 | 0.2239 | 93.63% | `418ea8ece74039b0783ff0b7c004a95f48c229a48d21246634956426c0918b6c` |
| `arm_a_replay25_lambda100_seed3030` | 25% | 1.00 | 3030 | 400 | 0.2165 | 93.54% | `fc373e502edd7ef50b17e2c0fbb94cb2835bb09dbb53f76c29b330ceb0117add` |
| `arm_a_replay25_lambda100_seed4040` | 25% | 1.00 | 4040 | 400 | 0.2324 | 93.23% | `9fd0c1a615acaf4843fe0008447d06d87998e09250386c5fdb2be3899cec023a` |
| `arm_b_replay50_lambda100_seed2026` | 50% | 1.00 | 2026 | 400 | 0.1988 | 94.68% | `fc0a6229c15bda9d464751df7fa44d706b00c5a979b94518390b6c662fcf33b6` |
| `arm_b_replay50_lambda100_seed3030` | 50% | 1.00 | 3030 | 400 | 0.2152 | 94.24% | `31384816671e4720c2ee896e61ca3b6398d4a3be1cc527911e37a5079a9bc67e` |
| `arm_b_replay50_lambda100_seed4040` | 50% | 1.00 | 4040 | 400 | 0.2074 | 94.66% | `9959ba7f2ee68240a6190d05a5395064f0939fea35c936d68a8a0dc459ccfd43` |
| `arm_c_replay25_lambda025_seed2026` | 25% | 0.25 | 2026 | 400 | 0.2357 | 92.96% | `88ffd5f4cc3a259d1bbab4965cc79ed7a732bcfc0731b75fd57a9fbac4b783f8` |
| `arm_c_replay25_lambda025_seed3030` | 25% | 0.25 | 3030 | 400 | 0.2359 | 93.36% | `0a78f59930377af6bfd7e210051ce38fdf5e14dfa2702d9168d4abae341674c6` |
| `arm_c_replay25_lambda025_seed4040` | 25% | 0.25 | 4040 | 400 | 0.2444 | 92.71% | `f758a089fa414a560a93fe1c8aae3c735e98c1dc99bb3c02ae4944134fb806cc` |
| `arm_d_replay50_lambda025_seed2026` | 50% | 0.25 | 2026 | 400 | 0.2161 | 93.92% | `76fed266dcc5ae3d59f9b792102c70433fca2392685f92ad5493a7cc8958d1a4` |
| `arm_d_replay50_lambda025_seed3030` | 50% | 0.25 | 3030 | 400 | 0.2118 | 94.14% | `8338cfb7bedbc351c1ee3c17f0885a8e1fadb23e53024dd4cac85fea4066a058` |
| `arm_d_replay50_lambda025_seed4040` | 50% | 0.25 | 4040 | 400 | 0.2466 | 93.04% | `97ad91ae9ebe4d5436b7f40cc693489bd077c6c0bd37131cb563bf5f6f984e65` |

---

## 3. Training Dynamics & Loss Trajectories

Across all 12 runs:
1. **Convergence:** Models converged smoothly from initial loss $\approx 1.0 - 1.3$ down to $0.10 - 0.24$ by step 400.
2. **Replay Loss:** The sequence-copying component loss $\mathcal{L}_{\text{copy}}$ decreased steadily from $0.6 - 1.4$ down to $0.03 - 0.15$, indicating that on the mini-batch replay examples, the model learned the training objective.
3. **Retrieval Loss:** Positive retrieval loss $\mathcal{L}_{\text{pos}}$ dropped from $1.8 - 2.5$ to $0.3 - 0.6$.
4. **Validation Loss & Accuracy:** Every model achieved best validation loss at step 400 with target token accuracy between 92.71% and 94.68%.

