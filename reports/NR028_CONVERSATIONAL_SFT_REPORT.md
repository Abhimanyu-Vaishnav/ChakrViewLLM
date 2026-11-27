# NR-028: Conversational Supervised Fine-Tuning (SFT) Report

## 1. Executive Summary

Phase 7 & 8 (**NR-027 / NR-028**) successfully completed conversational fine-tuning of ChakrGenesis-17M on the leak-free, deduplicated **ChakrView 1.0 Master Conversational Corpus (V1)** (`data/corpus_v1/`, 6,365 samples).

- **Base Checkpoint:** Canonical Stage-D Best (`experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`, SHA-256 `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`).
- **Target Release Checkpoint:** `experiments/genesis_stage_l_028/checkpoints/chakr_1_0_rc1_best.pt`
- **Trained Checkpoint SHA-256:** `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E`
- **Validation Loss:** Reached **0.9076** (Validation Perplexity **2.48**), achieving a significant improvement in fluency over Stage-D baseline.
- **Multilingual Dialogue:** Generates natural prose in English, Devanagari Hindi, and Hinglish.

---

## 2. Training Diagnostics & Convergence

| Training Parameter | Value |
| :--- | :--- |
| **Model Parameter Count** | 17,063,808 |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01) |
| **Learning Rate Schedule** | Cosine Annealing ($1\times 10^{-4} \rightarrow 1\times 10^{-5}$) |
| **Batch Size** | 16 (effective batch size with sequence length 256) |
| **Precision** | PyTorch Native FP16 Mixed Precision (`torch.amp`) |
| **Training Steps** | 730 steps across 2 epochs |
| **Total Training Duration** | 145.0 seconds (~2.4 minutes on RTX A1000 GPU) |
| **Epoch 1 Val Loss** | 1.0094 (PPL: 2.74) |
| **Epoch 2 Val Loss** | **0.9076** (PPL: **2.48**) |

---

## 3. Phase 8 Verdict
`PHASE 8 (NR-028) COMPLETE: CHAKRVIEW 1.0 RC1 NEURAL FOUNDATION CHECKPOINT FROZEN (SHA-256: CF585EBF...). CLEARED FOR COGNITIVE SUBSYSTEM INTEGRATION (PHASES 9–16).`
