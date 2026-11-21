# UCIP-06 Training Report

## 1. Training Setup & Architecture Lineage
- **ChakrMedium (12.93M parameters):**
  - Architecture: 7 Transformer layers, model width 384, 6 attention heads, hidden FFN dimension 896, max context 512, weight-tied embeddings.
  - Parent Checkpoint: `releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt` (`089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234`)
  - Candidate Best: `experiments/cognition/ucip_06/checkpoints/medium/candidate_best.pt`
  - Candidate Best SHA-256: `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72`

- **ChakrMicro (3.44M parameters):**
  - Architecture: 6 Transformer layers, model width 192, 6 attention heads, hidden FFN dimension 512, max context 512, weight-tied embeddings.
  - Parent Checkpoint: `releases/ucip_05_candidate/chakrview_ucip05_micro_model.pt` (`25933cc577a3c956d5359e9924eba22e403ef738df8bcac1880a212f8cb5b077`)
  - Candidate Best: `experiments/cognition/ucip_06/checkpoints/micro/candidate_best.pt`
  - Candidate Best SHA-256: `7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71`

---

## 2. Hyperparameters & Optimization
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98$, weight decay = $0.01$)
- **Learning Rate:** $3.0 \times 10^{-5}$
- **Batch Size:** 4 per micro-step
- **Gradient Accumulation:** 2 steps (Effective batch size = 8)
- **Gradient Clipping:** Max norm 1.0
- **Loss Masking:** Label `-100` on all prompt tokens; loss computed strictly on assistant continuation tokens.
- **Total Steps:** 300 steps per model.

---

## 3. Loss Dynamics

### ChakrMedium 12.93M
| Step | Train Loss | Validation Loss | Rollout Status |
|---|---|---|---|
| 0 (Parent Init) | — | 3.4318 | Memorized UCIP-05 templates |
| 25 | 3.4711 | 2.8676 | Rapid loss reduction |
| 50 | 2.8100 | 2.4940 | Beginning to adapt to new vocab distribution |
| 100 | 2.1185 | 2.2201 | Recovered exact copy ("The sky is blue") |
| 150 | 1.8807 | 1.9536 | English and Roman Hindi coherence stabilized |
| 200 | 1.5480 | 1.7294 | Devanagari Hindi generating grammatically sound tokens |
| 250 | 1.4360 | 1.5303 | Consistent conversational responses |
| **300 (Best)** | **1.0950** | **1.3709** | **Optimal validation loss achieved** |

### ChakrMicro 3.44M
| Step | Train Loss | Validation Loss | Rollout Status |
|---|---|---|---|
| 0 (Parent Init) | — | 3.2676 | Severe repetition loop ("neural neural...") |
| 25 | 3.7895 | 3.2676 | Escaping collapse basin |
| 50 | 3.1514 | 2.8680 | Basic English fragments emerging |
| 100 | 2.4126 | 2.5122 | Partial arithmetic tokens generated |
| 150 | 2.2875 | 2.3407 | Coherent phrase chunks |
| 200 | 1.9800 | 2.1888 | Devanagari words appearing |
| 250 | 2.0934 | 2.0842 | Improved vocabulary variety |
| **300 (Best)** | **1.7923** | **2.0039** | **Best validation checkpoint saved** |

---

## 4. Key Training Observations
1. **Convergence Behavior:** ChakrMedium reached a validation loss of 1.3709, smoothly decreasing from 3.4318. ChakrMicro reached 2.0039, showing significant capacity constraint compared to Medium.
2. **Loss Masking Verification:** Autoregressive rollout confirmed that the model generates clean completions directly after the `Assistant:` prompt token.
3. **Hardware Execution:** Pure CPU training on Windows ran at ~2 steps/sec on PyTorch CPU without out-of-memory or thread deadlocks.
