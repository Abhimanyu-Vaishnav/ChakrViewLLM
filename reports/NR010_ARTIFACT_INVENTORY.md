# CV-GENESIS-NR-010 Artifact Inventory

## 1. 17M Checkpoint Registry

| Checkpoint Identifier | File Path | Total Parameters | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **17M Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 17,063,808 | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` |
| **17M Pretrain Final**| `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_final.pt`| 17,063,808 | `c018e3221a09dab67e97ec2383def345652e3949893bab606f67e7d46cea0a3b` |
| **17M SFT Best** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | 17,063,808 | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` |
| **17M SFT Final** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_final.pt` | 17,063,808 | `c92c97265767a662936607d2342f513ad1a7a467427d6200de49cd2da86134aa` |

---

## 2. Code & Test Artifacts
- **Pretraining Script:** `experiments/genesis_pretraining_010/train_17m_pretraining.py`
- **Conversational SFT Trainer:** `experiments/genesis_pretraining_010/train_17m_sft.py`
- **300-Prompt Benchmark Runner:** `experiments/genesis_pretraining_010/eval_17m_vs_5m.py`
- **Comparative Metrics:** `experiments/genesis_pretraining_010/eval_300_comparison.json`
- **Training Logs:** `experiments/genesis_pretraining_010/training_log.json`
