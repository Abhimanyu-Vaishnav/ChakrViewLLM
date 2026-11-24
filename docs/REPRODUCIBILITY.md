# ChakrGenesis Reproducibility Guide

## 1. System Requirements

### Hardware Requirements
- **Training (Pretraining & SFT):**
  - Minimum: 1x NVIDIA GPU with $\ge$ 4 GB VRAM (e.g., GTX 1650, RTX 3050, T4).
  - Recommended: 1x NVIDIA GPU with $\ge$ 8 GB VRAM (RTX 3060/4060 or better).
  - CPU Training: Feasible but slow (~10–15x longer epoch time).
- **Inference:**
  - Can run on any modern x86_64 CPU (4 cores, 8 GB RAM) or NVIDIA/Apple Silicon GPU.
  - VRAM footprint during inference: **< 120 MB**.

### Software & Environment
- **Operating System:** Windows 10/11 or Linux (Ubuntu 20.04+).
- **Python:** 3.10 or 3.11 (Virtual environment required).
- **PyTorch:** $\ge 2.0.0$ (CUDA 11.8 or CUDA 12.1+).

---

## 2. Environment Setup

```powershell
# 1. Clone repository
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM

# 2. Check out working branch
git checkout exp/cbr-01-conversational-breakthrough

# 3. Pull Git LFS checkpoints
git lfs pull

# 4. Activate Python virtual environment
# Windows:
.\.venv_gpu\Scripts\activate
# Linux:
source .venv_gpu/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

---

## 3. Cryptographic Verification of Base Artifacts

Before launching training or evaluation, verify the integrity of the canonical artifacts:

```powershell
# PowerShell:
Get-FileHash data/experiments/vocab_4096/vocab.json -Algorithm SHA256
Get-FileHash experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt -Algorithm SHA256
Get-FileHash experiments/genesis_sft_005/checkpoints/genesis_sft_best.pt -Algorithm SHA256
```

Expected Hashes:
- `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- `canonical_pretrain_best.pt`: `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e`
- `genesis_sft_best.pt`: `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269`

---

## 4. Reproducing Canonical Pretraining (Mission NR-004)

```powershell
python -m torch.distributed.run --nproc_per_node=1 scripts/pretrain_canonical.py `
    --config experiments/genesis_canonical_pretraining_004/config.yaml `
    --vocab_path data/experiments/vocab_4096/vocab.json `
    --output_dir experiments/genesis_canonical_pretraining_004/reproduced_checkpoints/ `
    --seed 42
```

- **Batch Size:** 32
- **Learning Rate:** $3.0 \times 10^{-4}$ with Cosine Annealing to $3.0 \times 10^{-5}$
- **Epochs:** 5
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay = 0.1)

---

## 5. Reproducing Conversational SFT (Mission NR-005)

```powershell
python scripts/train_sft.py `
    --init_checkpoint experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt `
    --dataset_path data/experiments/genesis_sft_005/sft_balanced_manifest.jsonl `
    --vocab_path data/experiments/vocab_4096/vocab.json `
    --output_dir experiments/genesis_sft_005/reproduced_checkpoints/ `
    --batch_size 16 `
    --lr 5e-5 `
    --epochs 3 `
    --seed 1337
```

---

## 6. Running Independent Evaluation (Mission NR-006)

To execute the 300-prompt blind evaluation benchmark:

```powershell
python scripts/eval_genesis_pure_neural.py `
    --checkpoint experiments/genesis_sft_005/checkpoints/genesis_sft_best.pt `
    --vocab data/experiments/vocab_4096/vocab.json `
    --eval_set data/eval/nr006_blind_300.jsonl `
    --output reports/REPRODUCED_EVAL_RESULTS.json
```
