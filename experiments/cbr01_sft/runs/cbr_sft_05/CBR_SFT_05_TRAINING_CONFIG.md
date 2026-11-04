# CBR-SFT-05 Training Configuration & Execution Plan

**Experiment:** CBR-SFT-05 — Context Binding, Key-Value Induction & Conversational Memory  
**Base Lineage:** CBR-SFT-04 Step 200 (`experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`)  
**Parent Checkpoint SHA-256:** `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`  
**Run Directory:** `experiments/cbr01_sft/runs/cbr_sft_05/`  

---

## 1. Objectives & Architectural Context
- Train ChakrMicro (4 layers, 8 heads, d_model=512, d_head=64, context=512) on context-binding and key-value retrieval.
- Overcome the 0% context-binding and multi-turn retention ceiling observed in CBR-SFT-04.
- Prevent the overfitting and validation perplexity divergence observed in CBR-SFT-04 by regularizing learning rate and leveraging a 2.5× larger dataset.

---

## 2. Dataset Configuration
- **Training Set:** `data/sft/cbr_sft_05/splits/cbr_sft_05_train_360.jsonl` (360 examples)
- **Validation Set:** `data/sft/cbr_sft_05/splits/cbr_sft_05_val_40.jsonl` (40 examples)
- **Prompt Masking:** Strict assistant-only loss backpropagation (prompt tokens set to `-100`).
- **Context Length:** Max 512 tokens (actual max token length: 182 tokens).

---

## 3. Hyperparameters
- **Total Steps:** 300
- **Micro-Batch Size:** 2
- **Gradient Accumulation Steps:** 2 (Effective Batch Size: 4)
- **Peak Learning Rate:** $1.2 \times 10^{-4}$ (Conservative reduction from CBR-SFT-04's $2.0 \times 10^{-4}$)
- **Minimum Learning Rate:** $1.0 \times 10^{-5}$
- **LR Schedule:** Cosine decay with warmup
- **Weight Decay:** 0.01 (AdamW)
- **Gradient Clipping:** 1.0
- **Evaluation Interval:** Every 10 steps
- **Checkpoint Save Interval:** Every 50 steps (Step 50, 100, 150, 200, 250, 300)
- **Seed:** 42
- **Device:** CPU (deterministic execution)

---

## 4. Checkpoint Schedule
Checkpoints will be saved to `experiments/cbr01_sft/runs/cbr_sft_05/checkpoints/`:
- `sft_checkpoint_step_50.pt`
- `sft_checkpoint_step_100.pt`
- `sft_checkpoint_step_150.pt`
- `sft_checkpoint_step_200.pt`
- `sft_checkpoint_step_250.pt`
- `sft_checkpoint_step_300.pt`

