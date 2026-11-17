# UCIP-08: Training Execution & Optimization Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Run Name:** `ucip08_medium_recovery_01`  
**Base Architecture:** ChakrMedium (12,932,736 parameters, 7 layers, 384 dim, 6 heads, 896 ffn)  
**Parent Weights:** `releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt`  
**Date:** November 17, 2026  

---

## 1. Training Setup & Hyperparameters

- **Hardware Execution:** Local CPU-First Execution (Intel / AMD x86_64, Windows, Single Socket)
- **Precision:** Float32 (`torch.float32`)
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** $2.0 \times 10^{-5}$
- **Weight Decay:** $0.01$
- **Batch Size:** 4
- **Gradient Clipping:** Max norm $1.0$
- **Loss Function:** Label-Smoothed Cross-Entropy ($\epsilon = 0.05$) with target masking (`ignore_index = -100`)
- **Dataset Size:** 734 training samples, 82 validation samples
- **Total Steps:** 125 steps
- **Checkpointing:** Validated every 25 steps; best validation loss serialized

---

## 2. Loss & Progress Trajectory

| Step | Train Loss | Validation Loss | Wall-Clock Time | Checkpoint Saved? |
|---|---|---|---|---|
| 25 | 2.4301 | 1.9126 | 5.5s | Yes (`candidate_best.pt`) |
| 50 | 2.3598 | 1.7167 | 13.3s | Yes (`candidate_best.pt`) |
| 75 | 2.3395 | 1.6301 | 21.0s | Yes (`candidate_best.pt`) |
| 100 | 1.8890 | 1.5163 | 29.0s | Yes (`candidate_best.pt`) |
| 125 | 1.7450 | **1.4452** | 36.4s | Yes (`candidate_best.pt`) |

*Final Best Validation Loss:* **1.4452**  
*Total Training Time:* 36.4 seconds on CPU.

---

## 3. Qualitative Evolution During Training

### Greeting (`Hello! How are you?`)
- **Step 25:** `"Gou are very well is the ind morning! I am doing well and ready to assist you."`
- **Step 75:** `"I am ChakrView, your neural help you today?"`
- **Step 125:** `"Gou are very welp your neural help yow can I assist you?"`

### CPU Explanation (`What does a CPU do?`)
- **Step 25:** `"A CPU operates through the fetch-decode-execute cycle, retrieving instructions..."`
- **Step 50:** `"A CPU operates through the computer CPU operates through the ming calculates..."`
- **Step 125:** `"A CPU operates through the fetcheduses of a of a of a magnetic calculates..."`

### Arithmetic (`14 + 7`)
- **Step 25:** `17 + 7 = 17`
- **Step 50:** `18 + 7 = 17`
- **Step 75:** `12 + 7 = 27`
- **Step 125:** `14 + 7 = 12`

### Exact Copying (`Say exactly: The sky is blue.`)
- **Step 25:** `"The sky is blue and gold and pics blue and ps."`
- **Step 50:** `"The sky is blue and clear."`
- **Step 125:** `"The sky is blue and clear."`

---

## 4. Checkpoint Serialization & Integrity

- **Path:** `experiments/cognition/ucip_08/checkpoints/candidate_best.pt`
- **Release Path:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`
- **SHA-256 Digest:** `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631`
- **Size:** 51,756,113 bytes (51.8 MB)
- **Parameter Count:** 12,932,736
