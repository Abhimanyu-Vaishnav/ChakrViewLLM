# ChakrView CPT Preparation & Protocol Audit — Batches 2061–2100
**Milestone:** Batches 2061–2100  
**Scope:** Review of CONTROLLED_CPT_EXPERIMENT_PLAN.md, Operational Readiness, Validation Sufficiency & Release Gates  
**Date:** October 2026  
**Status:** Audit Complete  

---

## 1. Executive Summary

We audited the proposed continued-pretraining protocol set forth in `experiments/batch2021_2060/CONTROLLED_CPT_EXPERIMENT_PLAN.md` and `TRAINING_READINESS_DECISION.md`.

**CRITICAL POLICY ENFORCED:** Zero model training was executed during this audit.

---

## 2. Protocol Specification Verification

We evaluated the 14 core operational dimensions of the experiment plan:

| Dimension | Defined in Plan? | Audit Finding | Status |
|:---|:---:|:---|:---:|
| **1. Frozen Manifests** | Yes | `CORPUS_V2_MANIFEST.json` and `SPLIT_V2_MANIFEST.json` explicitly cited | **PASS** |
| **2. Untouched Final Test Protocol** | Yes | Absolute test split blindness enforced; early stopping on val only | **PASS** |
| **3. Checkpoint & Tokenizer Hashes** | Yes | Checkpoint SHA `96357705...` and `vocab_4096` verified | **PASS** |
| **4. Triplicate Seed Suite** | Yes | Seeds 42, 1337, and 2026 pre-registered | **PASS** |
| **5. Optimizer & Schedule** | Yes | AdamW ($\text{lr}=1.5 \times 10^{-4}$, cosine decay to $1.5 \times 10^{-5}$, 20 warmup steps) | **PASS** |
| **6. Batch & Context Length** | Yes | Batch size 4, grad accum 2 (eff batch 8), context $T=256$ | **PASS** |
| **7. Checkpoint Selection Rules** | Yes | Minimal val CE loss; patience = 3 checkpoints (75 steps) | **PASS** |
| **8. Resource Budget & Runtime** | Yes | CPU-first (Intel/AMD x86_64), $\le 2.0$ GB RSS, ~12–15 min/seed | **PASS** |
| **9. Resume & RNG State** | Partial | Script saves model state dict; optimizer & RNG state saving missing | **GAP IDENTIFIED** |
| **10. Loss Divergence / NaN** | Yes | Abort if val loss exceeds canonical $+0.25$ CE; check for NaNs | **PASS** |
| **11. Memory & Disk Limits** | Yes | Checkpoints saved every 25 steps (8 checkpoints $\times 42$ MB $\approx 336$ MB/seed) | **PASS** |
| **12. Capability Regression Tests** | Yes | Multi-domain evaluation across Math, Hindi, Sanskrit, Information Theory | **PASS** |
| **13. Deterministic Transfer Probes** | Yes | Arithmetic, Python AST syntax completion, factual cloze | **PASS** |
| **14. Predeclared Promotion Criteria**| Yes | Lower val CE without regression on individual test domains | **PASS** |

---

## 3. Analysis of Critical Operational Gaps & Remediation

### Gap 1: Validation Split Source Family Concentration
- **Observation:** In the proposed `SPLIT_V2_MANIFEST.json`, the Validation split contains 3 source families:
  1. `google_hinglish_top_dialogues`: 26,607 tokens (**81.57%** of validation tokens)
  2. `wikipedia_hi_ganit`: 4,741 tokens (**14.54%**)
  3. `cpython_algorithms_bisect`: 1,269 tokens (**3.89%**)
- **Risk:** Because Hinglish accounts for 81.6% of the validation tokens, early stopping and checkpoint selection decisions would be overwhelming driven by Hinglish dialogue loss, rather than a balanced multi-domain objective.
- **Remediation:** During validation checkpoint evaluation, the early-stopping metric must be calculated as an **equal-weighted macro average across the 3 validation source families**, rather than raw token-weighted average:
  $$\mathcal{L}_{\text{val}}^{\text{macro}} = \frac{1}{3} \left( \mathcal{L}_{\text{hinglish}} + \mathcal{L}_{\text{ganit}} + \mathcal{L}_{\text{bisect}} \right)$$

### Gap 2: Checkpoint State Serialization (Resume & RNG State)
- **Observation:** In existing CPT training code (`run_batch1861_experiments.py`), checkpoints were saved via `torch.save(m.state_dict(), ckpt_path)`, omitting optimizer state, step counter, and RNG seeds.
- **Remediation:** Phase 5 training runner must serialize a comprehensive dictionary:
  ```python
  torch.save({
      "step": step,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict(),
      "rng_state": torch.get_rng_state(),
      "seed": seed,
      "config": config,
  }, ckpt_path)
  ```

### Gap 3: Data Cleansing Pre-Flight Pass for Sanskrit
- **Observation:** Residual MediaWiki markup (`'''`, `''`, and indentation colons) is present in 49.2% of Sanskrit documents.
- **Remediation:** Incorporate regex markup stripping into the pre-flight dataset loader prior to executing the 200-step training loop.

---

## 4. Operational Release Gate Assessment

ChakrView's engineering assets are robust, reproducible, and mathematically coherent. However, because the validation split is 81.6% concentrated in a single source family and the Sanskrit raw extraction contains wiki markup residue, an unconditional `READY_FOR_CONTROLLED_CPT` would prematurely bypass necessary pre-flight safeguards.

Therefore, the scientifically defensible verdict is:
```
================================================================================
RELEASE DECISION: CONDITIONAL_CPT_READINESS
================================================================================
```
Training can proceed as soon as the pre-flight data cleaning and macro validation loss weighting are configured.
