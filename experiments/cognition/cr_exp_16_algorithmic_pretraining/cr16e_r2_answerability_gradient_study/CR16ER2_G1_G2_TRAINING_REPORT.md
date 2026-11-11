# CR-16E-R2 Phase E: Controlled Training Report (G1 vs G2)

### 1. Training Setup and Verification
A controlled 6-run experiment was conducted to directly test whether supervised gradients on negative abstention examples (`[UNKNOWN]`) causally degraded sequence copying or whether copying broke down due to other factors:
- **Parent Checkpoint:** CR-16A Arm D (`9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5`).
- **Dataset:** `arm_d_shuffled_balancedspan_train.json` ($4,000$ examples: $50\%$ retrieval, $37.5\%$ copying replay, $12.5\%$ abstention).
- **Seeds:** 2026, 3030, 4040.
- **Batch Size:** 16 ($250$ batches per epoch).
- **Steps:** Exactly 400 steps per run.
- **Condition G1:** Negative abstention loss enabled (standard training).
- **Condition G2:** Negative abstention loss masked (`labels = -100` on abstention examples).

---

### 2. Checkpoint Manifest & Cryptographic Hashes

| Run Name | Condition | Seed | Steps | Val Loss (Best) | Active Tokens Seen | Checkpoint File | SHA-256 Checksum |
|---|---|---|---|---|---|---|---|
| `arm_g1_enabled_seed2026` | G1 Enabled | 2026 | 400 | 0.2446 | 61,985 | `arm_g1_enabled_best.pt` | `0b6693915f1a532d85d5c11c136e6c13642c5ae0b2772793a42c81013a974b53` |
| `arm_g1_enabled_seed3030` | G1 Enabled | 3030 | 400 | 0.2204 | 61,616 | `arm_g1_enabled_best.pt` | `375ea414e7cc2ac90a7be0feaf661b26bc8a0b71987ef8eb3e05d272692494a1` |
| `arm_g1_enabled_seed4040` | G1 Enabled | 4040 | 400 | 0.2384 | 62,221 | `arm_g1_enabled_best.pt` | `55cf88c1fc16a98019af2b5645c46c69c63db16987ab02b503be2eb211ef7835` |
| `arm_g2_masked_seed2026` | G2 Masked | 2026 | 400 | 1.3952 | 53,885 | `arm_g2_masked_best.pt` | `08f82d2a5be713012cad1f891b8790aadfb3e7f3fd2a6a4541bf9f5daf36cfd9` |
| `arm_g2_masked_seed3030` | G2 Masked | 3030 | 400 | 1.3307 | 53,756 | `arm_g2_masked_best.pt` | `0f1e555b210085004aba6703926d981ee48fab380bd09d731901e96bba7a519e` |
| `arm_g2_masked_seed4040` | G2 Masked | 4040 | 400 | 1.4404 | 54,181 | `arm_g2_masked_best.pt` | `c3712ce3c833ef762f97d17c72268bc66b554fbf9200a6fe854900ef02a767fc` |

- Training logs and loss curves persisted in `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study/logs/`.

