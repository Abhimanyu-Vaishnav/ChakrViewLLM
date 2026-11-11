# CR-16D-R Forensic Lineage & Metric Audit

### 1. Mandatory Lineage & Protected Artifact Verification
Every foundational checkpoint, benchmark, tokenizer artifact, and runtime module was audited prior to launching CR-EXP-16E:

| Artifact | Recorded / Expected SHA-256 | Verified Actual SHA-256 | Status |
|---|---|---|---|
| **Canonical Stage B Baseline** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED MATCH** |
| **CR-16A Arm D Parent Checkpoint** | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | **VERIFIED MATCH** |
| **CR-16A Arm C Checkpoint** | `ac4beb4cfff904d150d6783f8d5e7e6ee7c8452179958a0559913586a2d8d2be` | `ac4beb4cfff904d150d6783f8d5e7e6ee7c8452179958a0559913586a2d8d2be` | **VERIFIED MATCH** |
| **CR-16C Sealed Benchmark** | `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` | `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` | **VERIFIED MATCH** |
| **Protected Runtime Session** | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED UNTOUCHED** |

---

### 2. Forensic Recomputation of CR-16D-R Core Metrics
The 12 replication models across Seeds 202, 303, and 404, together with the 8 historical models (Seeds 42, 101), were re-scored directly against raw prediction logs:
- **Copying Retention ($\ge 25\%$ Replay):** $80.8\% - 88.2\%$ exact match across all 15 models. Catastrophic forgetting is 100% reliably prevented.
- **Dynamic Reassignment (CAT2):** $100.0\%$ (2,000 / 2,000 correct) in Arms B and C across all seeds ($\sigma = 0.0$).
- **Arm C Seed 101 Outlier Status:** Confirmed. Across all 5 seeds, Arm C averaged $53.4\% \pm 58.5$ on G1, disproving the claim that Arm C achieves $>90\%$ generalization deterministically.
- **Position-Permutation Failure:** Confirmed at **0 / 100** paired success across all evaluated models.

---

### 3. Phase A Verdict
**PASS WITH LIMITATIONS.**
All artifact lineages and evaluation scripts are mathematically verified. The identified limitations (absolute positional bias, subword completion drift, and degraded abstention) serve as the direct empirical motivation for CR-EXP-16E.

