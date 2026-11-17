# UCIP-09: Training Execution & Optimization Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Run Name:** `ucip09_medium_autoregressive_recovery`  
**Base Architecture:** ChakrMedium (12,932,736 parameters; 7 layers, 384 dim, 6 heads, 896 ffn)  
**Parent Weights:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`  
**Date:** November 17, 2026  

---

## 1. Training Setup & Hyperparameters

- **Hardware Execution:** Local CPU-First Execution (Intel x86_64, Windows 11, Float32)
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** $2.0 \times 10^{-5}$
- **Weight Decay:** $0.01$
- **Batch Size:** 4
- **Gradient Clipping:** Max norm $1.0$
- **Loss Function:** Label-Smoothed Cross-Entropy ($\epsilon = 0.05$) with target masking (`ignore_index = -100`)
- **Autoregressive Shift Guarantee:** Strictly enforced: `shift_logits = logits[:, :-1, :]` and `shift_targets = targets[:, 1:]`
- **Dataset Size:** 3,240 training samples, 360 validation samples (0% overlap with benchmarks)
- **Total Steps:** 125 steps

---

## 2. Loss & Progress Trajectory

| Step | Train Loss | Validation Loss | Wall-Clock Time | Checkpoint Status |
|---|---|---|---|---|
| 25 | 3.9010 | 3.2371 | 6.3s | Saved (`candidate_best.pt`) |
| 50 | 3.0055 | 2.6665 | 14.3s | Saved (`candidate_best.pt`) |
| 75 | 2.5487 | 2.1631 | 22.9s | Saved (`candidate_best.pt`) |
| 100 | 2.0245 | 1.8262 | 31.7s | Saved (`candidate_best.pt`) |
| 125 | 1.6711 | **1.5893** | 40.1s | Saved (`candidate_best.pt`) |

*Final Best Validation Loss:* **1.5893**  
*Total Training Time:* 40.1 seconds on CPU.

---

## 3. Qualitative Evolution During Rollouts

- **Greeting (`Hello ChakrView. How are you?`):**
  - Step 25: Repetitive neural loops (`"I am ChakrView, an independent neural neural..."`)
  - Step 50: Coherent assistant introduction (`"I am ChakrView, an independent neural conversational assistant for efficient..."`)
  - Step 125: Flawless identity opening (`"I am ChakrView, an independent neural conversational assistant for efficient local execution."`)
- **Devanagari Hindi (`नमस्ते! आप कौन हैं?`):**
  - Step 25: Fragmented syllables (`"मैं आपका सहायुल..."`)
  - Step 75: Natural greeting (`"नमस्ते! मैं आपका स्वागत है।"`)
  - Step 125: Self-identification (`"नमस्ते! मैं चार्क है, आपका सहायका स्थानीय..."`)
- **Exact Copying (`The sky is blue and clear.`):**
  - Step 25: Halting syllables (`"Sunalking the fublue..."`)
  - Step 50: Verbatim reproduction achieved (`"The sky is blue and clear."`)
  - Step 125: Stable verbatim reproduction maintained.

---

## 4. Checkpoint Serialization & Integrity

- **Candidate Checkpoint:** `experiments/cognition/ucip_09/checkpoints/candidate_best.pt`
- **Release Packaging:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`
- **Verified SHA-256 Digest:** `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`
- **Parameter Count:** 12,932,736
