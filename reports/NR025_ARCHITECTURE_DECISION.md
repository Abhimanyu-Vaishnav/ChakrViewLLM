# NR-025: Neural Architecture Research & Release Selection Report

## 1. Executive Summary

Phase 5 (**NR-025**) evaluated the neural model architecture for **ChakrView 1.0 Release Candidate**.

Guided by NR-016 empirical evidence (which demonstrated that unguided 38M scaling offered 0 generalization advantage over 17M on low-resource corpora while drastically increasing VRAM requirements), this audit reviewed parameter efficiency, attention mechanisms, normalization, and hardware constraints.

### Architecture Selection Verdict:
**CONFIRM CHAKRGENESIS-17M AS THE CANONICAL CHAKRVIEW 1.0 FOUNDATION ARCHITECTURE.**
- **Parameter Count:** **17,063,808 parameters** (17.06M).
- **VRAM Footprint:** Base model occupies only **66.22 MB VRAM** in FP32 (~33 MB in FP16), and **359.64 MB** during sequence length 512 batch inference.
- **Hardware Margin:** Fits comfortably within the 6,144 MB VRAM budget of the local NVIDIA RTX A1000 GPU (< 6% VRAM utilization during inference, < 35% during batched training with gradient accumulation).
- **Architectural Modernity:** Grouped-Query Attention (GQA 4:1), SwiGLU non-linearities, Pre-RMSNorm, Rotary Positional Embeddings (RoPE), and untied output projections.

---

## 2. Frozen Release Candidate Architecture Specification

| Architectural Hyperparameter | Specification Value | Design Rationale |
| :--- | :---: | :--- |
| **Total Parameters** | `17,063,808` | Optimal capacity-to-data ratio for low-resource training. |
| **Vocabulary Size** | `4,096` | High compression byte-level BPE (8.94 bytes/token on Hindi). |
| **Number of Transformer Layers** | `10` | Depth provides hierarchical abstract feature extraction. |
| **Model Dimension ($d_{model}$)** | `384` | Balanced width ensuring gradient stability. |
| **Query Attention Heads ($n_q$)** | `12` | Head dimension = 32 ($384 / 12$). |
| **Key-Value Heads ($n_{kv}$)** | `3` | Grouped-Query Attention (4:1 ratio); reduces KV-cache by 75%. |
| **Feed-Forward Dimension ($d_{ffn}$)**| `1024` (SwiGLU) | Smooth gating non-linearity improving gradient flow. |
| **Normalization Type** | `RMSNorm` (Pre-Norm) | $\epsilon = 1e-5$; avoids mean centering overhead of LayerNorm. |
| **Positional Embeddings** | `RoPE` (Rotary) | $\theta = 10,000.0$; enables relative positional generalization. |
| **Max Sequence Context Length** | `512` | Standard context for interactive conversational turns. |
| **Tied Token Embeddings** | `False` | Independent output LM head for expressivity. |

---

## 3. Phase 5 Verdict
`PHASE 5 (NR-025) COMPLETE: CHAKRGENESIS-17M FROZEN AS CANONICAL FOUNDATION ARCHITECTURE. CLEARED FOR PHASE 6 (TRAINING OBJECTIVE & DATA-FLOW VALIDATION).`
