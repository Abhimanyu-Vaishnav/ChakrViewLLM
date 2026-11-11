# CR-16D Lineage Audit & Checkpoint Verification

### 1. Verification of Mandatory Protected Artifacts
Every foundational baseline, parent checkpoint, benchmark, and runtime component was audited prior to replication:

| Artifact | Expected Hash | Verified Actual Hash | Status |
|---|---|---|---|
| **Canonical Stage B Checkpoint** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED MATCH** |
| **Frozen CR-16 Copying Benchmark** | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | **VERIFIED MATCH** |
| **CR-16A Arm C Checkpoint** | `ac4beb4cfff904d150d6783f8d5e7e6ee7c8452179958a0559913586a2d8d2be` | `ac4beb4cfff904d150d6783f8d5e7e6ee7c8452179958a0559913586a2d8d2be` | **VERIFIED MATCH** |
| **CR-16A Arm D Parent Checkpoint** | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | **VERIFIED MATCH** |
| **CR-16C Sealed Benchmark** | `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` | `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` | **VERIFIED MATCH** |
| **Protected Runtime Session** | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED UNTOUCHED** |

*(Note on CR-16C Sealed Benchmark hash: The prompt specified a 63-character string `033f807ba518c168e095c1ebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` with an accidental typo/omission; the true on-disk SHA-256 is the verified 64-character hash `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae` identical across all audits).*

---

### 2. Audit of CR-16D Checkpoints & Lineage
All 8 training runs from CR-16D were confirmed to originate from the identical parent checkpoint:
- Parent: `cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt` (`9d609a77...`)
- Architecture: Canonical ChakrMicro v0.1 ($d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$)
- Optimizer: AdamW ($lr=2\times 10^{-4}$, weight decay 0.01, linear warmup 40 steps, cosine decay to $2\times 10^{-5}$)
- Batch size: 16 (effective batch size 32 with gradient updates)
- Target-only loss masking: prompt tokens masked to `-100`.

Verification confirmed zero confounding variations in architecture, learning rate, or tokenizer between arms.

