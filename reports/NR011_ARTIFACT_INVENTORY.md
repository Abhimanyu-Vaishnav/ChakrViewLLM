# CV-GENESIS-NR-011 Artifact Inventory

## 1. Checkpoint Registry

| Checkpoint Name | File Path | Parameters | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **NR-010 17M Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 17,063,808 | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` |
| **NR-010 17M SFT Best** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | 17,063,808 | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` |
| **NR-011 17M Stage-D Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 17,063,808 | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **NR-011 17M Stage-D Final** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_final.pt` | 17,063,808 | `bcca0c323ee2fc5c3f159ed5588af8fd195c11b1f76663f2d2d6d69c2274c2fe` |

---

## 2. Dataset & Benchmark Manifests
- **Stage-D Corpus Splits:** `data/stage_d/splits/` (4,686 train, 551 val, 276 test)
- **Stage-D Manifest:** `data/stage_d/stage_d_manifest.json`
- **Sealed Benchmark Suite:** `data/sealed_eval_011/sealed_benchmark_prompts.json`
- **Sealed Evaluation Results:** `experiments/genesis_stage_d_011/evaluations/sealed_eval_results.json`
