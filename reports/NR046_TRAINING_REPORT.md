# CHAKRVIEW NR-046: RC2 CANDIDATE TRAINING REPORT

**Mission Reference:** `CHAKRVIEW NR-046`  
**Phase:** WP-3 (RC2 Candidate SFT Training)  
**Date:** 2026-11-27  
**Artifact Directory:** `experiments/genesis_stage_rc2_046/`  
**Candidate Checkpoint:** `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt`  

---

## 1. Executive Summary

Under **WP-3 of NR-046**, a new candidate checkpoint (**ChakrView 1.0 RC2**) was trained using the verified response-masked loss supervision on Corpus V2.

### Non-Negotiable Boundary Compliance:
1. **Historical Checkpoints Preserved:**
   - Canonical Stage-D (`genesis_17m_staged_best.pt`, SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`): Intact and unperturbed.
   - Release Candidate 1 (`chakr_1_0_rc1_best.pt`, SHA-256: `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e`): Intact and frozen.
2. **Initialization:** Model weights were initialized strictly from the canonical Stage-D best checkpoint (`2c21632e...`).
3. **Tokenizer:** Canonical Byte-Level BPE 4096 tokenizer frozen and used without modification.

---

## 2. Training Configuration & Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Model Architecture** | `ChakrGenesis17M` (18.6M parameters) | Pretrained baseline architecture |
| **Initialization Base** | `genesis_17m_staged_best.pt` | Clean canonical pretraining checkpoint |
| **Dataset** | Corpus V2 (`data/corpus_v2/corpus_v2_train.jsonl`) | Cleaned prompts, balanced copy tasks |
| **Dataset Size** | 5,089 Train / 435 Validation | Deduplicated, leakage-free |
| **Batch Size** | 16 | Optimal for 6GB VRAM |
| **Max Sequence Length** | 256 | RoPE dynamic frequency caching enabled |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay=0.01) | Standard stable optimizer |
| **Learning Rate Schedule** | Cosine Annealing ($1.0 \times 10^{-4} \to 1.0 \times 10^{-5}$) | Smooth decay over 638 steps |
| **Loss Objective** | Response-Only CrossEntropy (`ignore_index=-100`) | Prompt tokens excluded from gradients |
| **Gradient Clipping** | 1.0 | Stability against gradient spikes |
| **Mixed Precision** | FP16 AMP | Accelerated GPU throughput on RTX A1000 |
| **Epochs** | 2 | Prevent overfitting on 17M capacity |
| **Total Training Time** | 6.55 minutes | Efficient local GPU execution |

---

## 3. Training & Validation Trajectory

| Epoch | Step | Train Loss (Response-Only) | Validation Loss (Response-Only) | Validation Perplexity | Best Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0 (Stage-D Zero-Shot) | N/A | **3.4154** | **30.43** | Base checkpoint |
| 1 | 50 | 1.9861 | 1.9119 | 6.77 | Yes |
| 1 | 100 | 1.7444 | 1.8125 | 6.13 | Yes |
| 1 | 150 | 1.6163 | 1.7407 | 5.70 | Yes |
| 1 | 200 | 1.5245 | 1.6930 | 5.44 | Yes |
| 1 | 250 | 1.4573 | 1.6504 | 5.21 | Yes |
| 1 | 300 | 1.4026 | 1.6298 | 5.10 | Yes |
| 1 | 319 | 1.3860 | 1.6196 | 5.05 | Yes |
| 2 | 50 | 0.9457 | 1.6014 | 4.96 | Yes |
| 2 | 100 | 0.8685 | 1.5830 | 4.87 | Yes |
| 2 | 150 | 0.8547 | 1.5640 | 4.78 | Yes |
| 2 | 200 | 0.8484 | 1.5581 | 4.75 | Yes |
| 2 | 250 | 0.8454 | 1.5459 | 4.69 | Yes |
| 2 | 300 | 0.8360 | **1.5382** | **4.66** | **Yes (Best)** |
| 2 | 319 | 0.8362 | 1.5384 | 4.66 | No |

---

## 4. Key Checkpoint Hashes & Verification

| Checkpoint Name | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **ChakrView 1.0 RC2 Best** | `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` | `f5ddb9a5c88c7553b6f0e95ffcf02830833a69a4563ff4e20790d565780eb7c7` |
| **Stage-D Base Initializer** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **Historical RC1 Checkpoint** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e` |

Validation loss decreased monotonically from $3.4154$ (Perplexity 30.43) to **1.5382** (Perplexity 4.66), confirming steady optimization without gradient explosions.
