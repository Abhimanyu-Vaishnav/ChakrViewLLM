# CR-EXP-21: Training Report — ChakrView v0.11

**Experiment ID:** CR-EXP-21  
**Candidate Model:** ChakrView Conversational Model v0.11  
**Checkpoint Path:** `experiments/cognition/cr_exp_21/training/v011_candidate_best.pt`  
**Parent Checkpoint:** `releases/v0.10.1_conversational_model/chakrview_v0101_model.pt`  
**Date:** 2026-11-15  
**Device:** CPU (PyTorch 2.6.0+cpu, Windows 11, Intel Core i7)  

---

## 1. Training Configuration & Hyperparameters

The training execution utilized the frozen canonical ChakrMicro v0.1 architecture:
- **Total Parameters:** $3,443,136$
- **Hidden Dimension:** 192
- **Transformer Layers:** 6
- **Attention Heads:** 6 (Head dimension: 32)
- **Feedforward Dimension:** 512 (SwiGLU activation)
- **Position Embedding:** Rotary Position Embedding (RoPE)
- **Normalization:** RMSNorm
- **Vocabulary Size:** 4,096 tokens (Chakr-BPE)
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98, \text{weight\_decay}=0.01$)
- **Learning Rate:** $1.5 \times 10^{-4}$ with linear warmup (20 steps) and cosine decay
- **Batch Size:** 4 dialogues per batch with gradient accumulation step of 2 (effective batch size: 8)
- **Max Sequence Length:** 256 tokens per dialogue
- **Total Steps:** 400 steps
- **Loss Masking:** Supervised Fine-Tuning (SFT) masking; loss computed strictly on Assistant response tokens.

---

## 2. Loss & Convergence Trajectory

| Step | Training Loss | Validation Loss | Token Accuracy | Learning Rate |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2.5126 | 2.5084 | 55.42% | $7.50 \times 10^{-6}$ |
| 50 | 2.3810 | 2.3615 | 58.10% | $1.42 \times 10^{-4}$ |
| 100 | 2.2945 | 2.2891 | 59.45% | $1.28 \times 10^{-4}$ |
| 150 | 2.2412 | 2.2350 | 60.12% | $1.09 \times 10^{-4}$ |
| 200 | 2.1980 | 2.2045 | 60.78% | $8.75 \times 10^{-5}$ |
| 250 | 2.1654 | 2.1890 | 61.05% | $6.50 \times 10^{-5}$ |
| 300 | 2.1420 | 2.1795 | 61.18% | $4.25 \times 10^{-5}$ |
| **350** | **2.1280** | **2.1737** | **61.26%** | $2.30 \times 10^{-5}$ |
| 400 | 2.1215 | 2.1742 | 61.24% | $1.00 \times 10^{-5}$ |

- **Best Validation Step:** Step 350
- **Best Validation Loss:** **2.1737**
- **Token Accuracy:** **61.26%**
- **Comparison to Historical Baselines:**
  - CR-20 Best Val Loss: $2.3196$
  - CR-20R Best Val Loss: $2.7878$
  - CR-21 v0.11 Best Val Loss: **2.1737** (Substantial cross-entropy improvement across the expanded 100+ entity domain)

---

## 3. Checkpoint Artifacts & Hashes

- **Candidate Checkpoint:** `experiments/cognition/cr_exp_21/training/v011_candidate_best.pt`
  - File Size: 41,388,167 bytes
  - SHA-256: `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754`
- **Release Checkpoint Copy:** `releases/v0.11_conversational_model/chakrview_v011_model.pt`
  - Verified Identical SHA-256: `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754`

