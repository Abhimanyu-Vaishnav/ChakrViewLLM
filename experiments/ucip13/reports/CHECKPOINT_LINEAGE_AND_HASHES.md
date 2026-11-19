# UCIP-13: Checkpoint Lineage & Hashes Report

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Parent Initialization:** Deterministic Random Initialization (seed 42, `c5571c9c...`)  
**Training Pipeline:** `train_stage_c.py` on Stage C Train  

---

## 1. Checkpoint Lineage Table

| Checkpoint Identifier | Global Step | Cumulative Tokens Exposed | Validation Loss | Validation Perplexity | SHA-256 Hash | Status |
|---|---|---|---|---|---|---|
| `checkpoint_step_000.pt` | 0 | 0 | 8.3084 | 4,057.62 | `e00d2b4bb215c453ec60a9f91df5791b879a92a58daba3ff8ce633408ed1a2c4` | Initial State |
| `checkpoint_step_050.pt` | 50 | 409,600 | 7.1541 | 1,279.38 | `58966b1f3ce34c76d670c471b894a643cd1b7c75d37ff68da9292acc5f09e36e` | Validated |
| `checkpoint_step_100.pt` | 100 | 819,200 | 6.6239 | 752.88 | `29e67df0af10f2a39172bf20af412e97734df006ae21e6b2d9c9686140e1e1be` | Validated |
| `checkpoint_step_200.pt` | 200 | 1,638,400 | 6.3261 | 558.96 | `f916c7bfb915d8e234ba479a9addc1af747e3da456f8a73b4adc89107702e485` | Validated |
| `checkpoint_step_400.pt` | 400 | 3,276,800 | 5.0917 | 162.67 | `488ec961437bc85abb00627a1aebe67e4140601e4ae7600e9a219a51bf7654bb` | Validated |
| `checkpoint_step_600.pt` | 600 | 4,915,200 | 4.7660 | 117.45 | `8fb6735d31af9e02f5ec16536a0b0a006132fac531f25e932eac78e253abc7fb` | Validated |
| `checkpoint_step_800.pt` | 800 | 6,553,600 | 4.6859 | 108.41 | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **BEST FINAL** |

All checkpoints are preserved on disk and recorded in `experiments/ucip13/checkpoints/checkpoint_manifest.json`.
