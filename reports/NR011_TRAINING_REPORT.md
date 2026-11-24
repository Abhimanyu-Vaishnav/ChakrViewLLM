# CV-GENESIS-NR-011 17M Stage-D Training Report

## 1. Executive Summary

Mission **CV-GENESIS-NR-011 Phase 5** completed conversational fine-tuning of `ChakrGenesis-17M` on the Stage-D multilingual corpus (`data/stage_d/splits/stage_d_train.jsonl`).

The candidate was warm-started directly from the verified 17M foundation checkpoint `genesis_17m_pretrain_best.pt` (`96219f22...`).

---

## 2. Training Dynamics & Loss Convergence

- **Run ID:** `genesis_17m_staged_011`
- **Total Parameters:** **17,063,808**
- **Hardware:** NVIDIA RTX A1000 Laptop GPU (6.0 GB VRAM)
- **Batch Size:** 16 ($B=16, T=256$, effective batch 4,096 tokens)
- **Total Steps:** 586 steps across 2 epochs (104.9 seconds total runtime)
- **Optimizer:** AdamW (learning rate $6.0 \times 10^{-5}$ with cosine decay to $6.0 \times 10^{-6}$)

### Loss & Perplexity Trajectory:
- **Initialization (Pretrained Zero-Shot on Stage-D):**
  - Validation Loss: **4.5853** | Validation Perplexity: **98.03**
- **Epoch 1 Summary:**
  - Validation Loss: **2.1012** | Validation Perplexity: **8.18**
- **Epoch 2 Summary (Convergence):**
  - Validation Loss: **1.9155** | Validation Perplexity: **6.79**

**Total Loss Drop:** From $4.5853$ to **$1.9155$** (Perplexity dropped by **$93.1\%$** from $98.03$ to **$6.79$**), establishing the lowest validation perplexity achieved across any model in ChakrView history.
