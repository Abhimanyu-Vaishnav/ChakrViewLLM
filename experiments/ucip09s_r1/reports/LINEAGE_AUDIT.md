# Stage 2 — Lineage and Configuration Audit Report

**Date:** 2026-11-18  
**Audit Mission:** UCIP-09S-R1 Independent Evidence Verification & Evaluator Audit  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Current HEAD:** `d9316f4a7823191f6eea1e245c891163db4392d9`  
**Preflight Reference HEAD:** `32100acd731256fc785f7cf043a92444fec32fd6`

---

## 1. Resolution of the Git HEAD Discrepancy

In the UCIP-09S preflight report, HEAD was recorded as `32100ac`. The reported completion commit was `d9316f4`.

### Verification via Git Log and Reflog:
- `git log -n 2 --oneline` shows:
  ```text
  d9316f4 feat(ucip09s): complete controlled retention and capability isolation study
  32100ac feat(cognition): complete UCIP-09R loss mathematics verification and recovery audit
  ```
- **Finding:** There is **no discrepancy or missing commit history**. Commit `32100ac` was the immediate parent commit of `d9316f4`. When UCIP-09S began, the working tree was checked out at `32100ac`. Upon completing the study deliverables and committing them, the new HEAD became `d9316f4`. All changes in `d9316f4` are strictly additions under `experiments/ucip09s/` (20 files, 3,734 insertions).

---

## 2. Checkpoint Ancestry & Parameter Lineage

The complete checkpoint chain was verified through metadata extraction and state dict analysis:

```
[UCIP-07 Medium]
       │
       ▼
[UCIP-08 Medium Candidate] (fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631)
  - Trained in: experiments/cognition/ucip_08/train_ucip08.py
  - Dataset: experiments/cognition/ucip_08/data/ucip08_train.jsonl (734 samples)
       │
       ▼
[UCIP-09 Medium Candidate] (5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499)
  - Parent in metadata: chakrview_ucip08_medium_model.pt
  - Trained in: experiments/cognition/ucip_09/train_ucip09.py
  - Dataset: experiments/cognition/ucip_09/data/ucip09_train.jsonl (3,240 samples across 7 balanced categories)
  - Optimizer steps: 125 steps, Batch size: 4, LR: 2e-5, Label smoothing: 0.05
  - Val loss achieved: 1.5893
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
[UCIP-09R Medium] (e1b943222...)            [UCIP-09S Condition C Pilot] (c99dcc0a2...)
  - Parent: chakrview_ucip09_medium_model.pt   - Parent: chakrview_ucip09_medium_model.pt
  - Trained in: train_ucip09r.py               - Trained in: train_ucip09s_pilot.py
  - Dataset: ucip09r_train.jsonl (885 samples) - Dataset: ucip09s_balanced_train.jsonl (1,080 samples)
  - Steps: 160 steps, Batch size: 4            - Steps: 80 steps, Batch size: 4
  - LR: 2.5e-5, Label smoothing: 0.03          - LR: 1.8e-5, Label smoothing: 0.05
  - Val loss achieved: 1.2461 (on 88% math)    - Val loss achieved: 1.6461 (on balanced split)
```

---

## 3. Condition C Pilot Training Configuration Verification

From `experiments/ucip09s/scripts/train_ucip09s_pilot.py` and the raw training console logs:

| Hyperparameter / Parameter | Target Specification | Actual Training Implementation | Verified |
|---|:---:|:---:|:---:|
| **Starting Checkpoint** | UCIP-09 Medium | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | Yes (SHA-256: `5513fefe...`) |
| **Output Checkpoint** | Condition C Pilot | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | Yes (SHA-256: `c99dcc0a...`) |
| **Optimizer** | AdamW | `AdamW(model.parameters(), lr=1.8e-5, weight_decay=0.01)` | Yes |
| **Learning Rate** | $1.8 \times 10^{-5}$ | $1.8 \times 10^{-5}$ (Constant schedule) | Yes |
| **Label Smoothing** | 0.05 | `nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)` | Yes |
| **Gradient Clipping** | 1.0 | `clip_grad_norm_(model.parameters(), 1.0)` | Yes |
| **Batch Size** | 4 | 4 | Yes |
| **Total Steps** | 80 | 80 | Yes |
| **Total Sample Exposures** | 320 exposures | 80 steps $\times$ 4 per step = 320 exposures | Yes |
| **Checkpoint Selection Logic** | Min val loss on balanced val set | Best val loss evaluated every 20 steps over 32 validation batches | Yes (Saved at step 80 with loss 1.6461) |

---

## 4. Actual Consumed Category Distribution During Condition C Training

The pilot dataset `ucip09s_balanced_train.jsonl` contains 1,080 samples.
Across the 80 steps (batch size 4, seed 42), the model consumed **320 sample exposures**:

| Category | Consumed Exposures | Consumed Percentage | Intended Mixture Target |
|---|:---:|:---:|:---:|
| **English conversation** | 71 | 22.2% | ~20% |
| **Devanagari Hindi** | 65 | 20.3% | ~20% |
| **Arithmetic** | 55 | 17.2% | ~20% |
| **Instruction following** | 49 | 15.3% | ~15% |
| **English explanations** | 28 | 8.8% | ~8% |
| **Hinglish** | 26 | 8.1% | ~8% |
| **Exact copying** | 26 | 8.1% | ~9% |
| **Total Anchor Replay (Conversational / Multi / Inst)** | **239** | **74.7%** | **~70.0%** |
| **Total Targeted Capabilities (Arithmetic + Copying)** | **81** | **25.3%** | **~30.0%** |

### Finding:
The sampling strategy accurately delivered the declared Condition C balanced replay target: **74.7% anchor replay vs 25.3% targeted capabilities**, preventing the massive 88.5% arithmetic skew of UCIP-09R.
