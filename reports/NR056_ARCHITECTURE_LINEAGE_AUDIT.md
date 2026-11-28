# NR-056: ARCHITECTURE LINEAGE & VARIANT AUDIT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Investigation of historical 17M architecture variants, resolution of Stage-D configuration documentation discrepancies, and verification of 38M architecture parameters across all checkpoints.

---

## 1. Executive Summary

Mission NR-056 conducted a rigorous, byte-level and tensor-level investigation into the architecture of all 12 primary ChakrView checkpoints, specifically addressing:
1. **The 17M Architecture Variant Question:** Earlier canonical Stage-D records occasionally cited `d_model=512, 6 layers, 8 heads` or `d_model=192, 6 layers`. This audit investigated whether multiple 17M architecture variants were ever physically instantiated into checkpoints.
2. **The 38M Architecture Verification:** Re-verifying all four 38M checkpoints (`NR-051 Base/SFT` and `NR-052 Base/SFT`) against the previously debunked `d_model=768, 6 layers` citation.

---

## 2. Key Findings: 17M Architecture Variant Investigation

### A. Tensor Reality of Stage-D Canonical Checkpoint
Inspection of `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` (`2c21632e...`) reveals:
- **Embedding:** `embedding.weight` has shape `[4096, 384]`.
- **Layers:** Exactly 10 transformer blocks (`layers.0` through `layers.9`).
- **Attention Projections:**
  - $W_{\text{q}}$: `[384, 384]` (12 query heads $\times$ 32 head dimension)
  - $W_{\text{k}}$: `[96, 384]` (3 key heads $\times$ 32 head dimension, GQA 4:1)
  - $W_{\text{v}}$: `[96, 384]` (3 value heads $\times$ 32 head dimension, GQA 4:1)
  - $W_{\text{o}}$: `[384, 384]`
- **Feed-Forward:** SwiGLU `[1024, 384]` intermediate dimension.
- **Embedded Config Object:** Class `chakrview.brain.genesis.chakr_genesis_17m.Genesis17MConfig` with exact fields: `d_model=384, n_layers=10, n_q_heads=12, n_kv_heads=3, hidden_dim=1024`.

### B. Provenance of the `d_model=512, 6 layers` and `d_model=192, 6 layers` Citations
- **`d_model=192, 6 layers` (3.44M params):** Refers to **ChakrMicro v0.1** (`chakrview/brain/model.py`), an early architectural prototype developed in early pre-Genesis research waves (waves 470–530) for unit-testing the cognitive orchestration runtime. It was never a 17M model.
- **`d_model=512, 6 layers, 8 heads` (~17.5M params):** Refers to an initial paper design / proposal documented in early planning specs. However, when `ChakrGenesis17M` was implemented and trained in Mission CV-GENESIS-NR-009/011, the architecture was explicitly built as a 10-layer GQA network (`d=384, L=10, Q=12, KV=3`) to optimize inference cache throughput.
- **Verdict:** There is **only one physical 17M architecture in existence** across all checkpoints: `ChakrGenesis17M` (`d_model=384, L=10`). Every historical 17M checkpoint—Stage-D Canonical, RC1, RC2, RC3, NR-051 17M Base/SFT, and NR-052 17M Base/SFT—is identical in tensor structure, parameter count (17,063,808 unique params), and configuration class. The `d=512, L=6` description in earlier text was a **documentation artifact from an early proposal**, not an alternative checkpoint family.

---

## 3. Key Findings: 38M Architecture Verification

All four 38M checkpoints (`chakr_38m_pretrained_base.pt`, `chakr_38m_sft_candidate.pt`, `chakr_38m_v4_base.pt`, and `chakr_38m_nr052_sft.pt`) were independently re-verified:
- **Embedding:** `[4096, 512]`
- **Layers:** Exactly 12 transformer blocks (`layers.0` through `layers.11`)
- **Attention Projections:** $W_{\text{q}}$ `[512, 512]` (8 Q heads $\times$ 64 head dim), $W_{\text{k}}$ `[256, 512]` (4 KV heads $\times$ 64 head dim, GQA 2:1)
- **Feed-Forward:** SwiGLU `[1408, 512]` intermediate dimension
- **Unique Trainable Parameters:** **37,499,392**
- **Verdict:** The `d_model=768, 6 layers` entry in earlier documents was an erroneous transcription from an abandoned scaling hypothesis. All trained 38M artifacts strictly conform to `d_model=512, L=12`.

---

## 4. Reconciled Architectural Lineage Summary

```
                ┌────────────────────────────────────────────────────────┐
                │          Foundational Architectural Families           │
                └───────────────────────────┬────────────────────────────┘
                                            │
               ┌────────────────────────────┴───────────────────────────┐
               ▼                                                        ▼
    ChakrGenesis-17M Family                                  ChakrGenesis-38M Family
    - d_model = 384, L = 10                                  - d_model = 512, L = 12
    - Q = 12, KV = 3 (GQA 4:1)                               - Q = 8, KV = 4 (GQA 2:1)
    - hidden_dim = 1024 (SwiGLU)                             - hidden_dim = 1408 (SwiGLU)
    - Unique Params: 17,063,808                              - Unique Params: 37,499,392
    - Checkpoints: Stage-D, RC1, RC2,                        - Checkpoints: NR-051 Base/SFT,
      RC3, NR-051 17M, NR-052 17M                              NR-052 Base/SFT
```
