# NR-052: Controlled Pretraining Results on Corpus V4

**Mission:** CHAKRVIEW NR-052  
**Date:** November 27, 2026  
**Artifact Directory:** `experiments/pretraining_nr052/`  
**Execution Summary:** [NR052_PRETRAINING_SUMMARY.json](file:///D:/Project/ChakrView/experiments/pretraining_nr052/NR052_PRETRAINING_SUMMARY.json)

---

## 1. Executive Summary

Both models were pretrained from scratch (`seed=42`) on the verified **ChakrView Corpus V4** under strictly matched training conditions (batch size 32, max sequence length 256, 358 total steps, 1,181,206 tokens seen).

```mermaid
graph LR
    C4[Corpus V4<br>624k Clean Tokens] --> PT17[ChakrGenesis-17M<br>Best Val Loss: 1.4763<br>PPL: 4.38 | Acc: 74.8%]
    C4 --> PT38[ChakrGenesis-38M<br>Best Val Loss: 1.1684<br>PPL: 3.22 | Acc: 78.9%]
```

---

## 2. Quantitative Comparison: 17M vs 38M on Corpus V4

| Metric | ChakrGenesis-17M Base | ChakrGenesis-38M Base | Delta (38M vs 17M) | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Unique Parameters** | 17,063,808 | 37,499,392 | +20,435,584 (+120%) | Scaled capacity |
| **Tokens Seen** | 1,181,206 | 1,181,206 | 0 | Matched budget |
| **Initial Val Loss (Step 75)**| 4.7140 (PPL 111.5) | 4.3012 (PPL 73.8) | -0.4128 nats | Faster initial acquisition |
| **Mid Val Loss (Step 225)** | 1.9071 (PPL 6.73) | 1.6055 (PPL 4.98) | -0.3016 nats | Sustained convergence |
| **Best Final Val Loss** | **1.4763** | **1.1684** | **-0.3079 nats** | **-20.9% loss reduction** |
| **Best Perplexity (PPL)** | **4.38** | **3.22** | **-1.16 PPL** | **-26.5% perplexity reduction** |
| **Next-Token Accuracy** | **74.8%** | **78.9%** | **+4.1%** | Substantial accuracy lead |
| **Training Duration** | 93.94 s (1.57 min) | 132.73 s (2.21 min) | +38.79 s | High efficiency |
| **Peak GPU VRAM** | 2,795.3 MB | 3,639.7 MB | +844.4 MB | Well within 6GB limit |
| **Checkpoint Path** | `chakr_17m_v4_base.pt` | `chakr_38m_v4_base.pt` | — | — |

---

## 3. Comparison with NR-051 Pretraining

Cleaning the dataset (eliminating template multipliers and boilerplate) produced a dramatic leap in pretraining convergence across both architectures:
- **17M Architecture:** NR-051 Val Loss: 3.0516 (PPL 21.15, Acc 53.2%) $\to$ **NR-052 Val Loss: 1.4763 (PPL 4.38, Acc 74.8%)**.
- **38M Architecture:** NR-051 Val Loss: 2.8502 (PPL 17.29, Acc 55.4%) $\to$ **NR-052 Val Loss: 1.1684 (PPL 3.22, Acc 78.9%)**.

Data quality filtering removed noise and enabled the causal transformer models to learn stable grammatical and semantic representations.
