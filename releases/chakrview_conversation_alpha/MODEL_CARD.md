# ChakrView Model Card: Conversation Alpha

## Model Overview
- **Model Name:** ChakrView Conversation Alpha
- **Release Directory:** `releases/chakrview_conversation_alpha/`
- **Checkpoint File:** `chakrview_conversation_alpha.pt`
- **Architecture:** ChakrMicro v0.1 (Decoder-only Causal Transformer, RoPE, RMSNorm, SwiGLU, Tied Weights)
- **Parameters:** 3,443,136 unique parameters
- **Context Length:** 512 tokens
- **Vocabulary:** 4,096 tokens (BPE)
- **Checkpoint SHA-256:** `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7`

---

## Training Lineage & Methodology

1. **Foundational Pre-Training (UCIP-13):**
   - Trained from scratch on 27 shards of the clean Stage C multilingual corpus (6.55M tokens).
   - Validation Loss: 4.6859.
2. **Supervised Fine-Tuning (UCIP-15):**
   - Trained using an 80/20 mixture: 12,000 instruction-response records from `data/sft/ucip14r/` combined with continuous Stage C replay streams to eliminate catastrophic forgetting.
   - Optimization: AdamW, learning rate $2.5 \times 10^{-5}$ decaying to $2.5 \times 10^{-6}$, 50-step warmup, token-normalized joint loss.
   - SFT Validation Loss: Dropped from 5.5442 to 3.8426 (Step 550).
   - Foundational Retention Loss: Preserved at 3.6887 (better than pretraining baseline 3.9544).

---

## Behavioral Evaluation Summary (300 Non-Sealed Dev Probes)

- **Strict Aggregate Score:** 70.33% (vs. 37.0% base, **+33.3% absolute gain**)
- **Repetition Rate:** Reduced from 54.0% down to **3.33%**
- **English Adherence:** 75.32%
- **Hinglish Adherence:** 90.32%
- **Hindi Adherence:** 62.24%
- **Zero Sealed Benchmark Contamination:** The 500 quarantined sealed benchmark probes remained completely untouched throughout training and model selection.
