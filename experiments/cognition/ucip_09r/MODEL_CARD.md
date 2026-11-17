# ChakrView UCIP-09R Medium Candidate Model Card

## Model Details
- **Model Name:** ChakrView Medium (UCIP-09R Candidate)
- **Architecture Family:** Indigenous Causal Transformer Decoder
- **Parameter Count:** 12,932,736
- **Layers:** 7
- **Model Width ($d_{\text{model}}$):** 384
- **Attention Heads:** 6
- **Feed-Forward Dimension ($d_{\text{ff}}$):** 896
- **Vocabulary Size:** 4,096 byte-level BPE
- **Positional Encoding:** RoPE ($\theta = 50,000.0$)
- **Normalization:** Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Tied Embeddings:** True
- **Serialization Path:** `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt`
- **SHA-256 Digest:** `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0`
- **Parent Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` (`5513fefe...`)

---

## Evaluation Benchmark Performance
- **Sealed Language Intelligence 500:** **11.4%** (57 / 500)
  - English conversation: 7.1%
  - Devanagari Hindi: 40.0%
  - Hinglish: 18.6%
  - Semantic consistency: 32.0%
- **Owner Reality 100:** **8.0%** (8 / 100)

---

## Technical Finding
Fine-tuning specifically on a smaller natural arithmetic corpus improved Devanagari Hindi (40.0%) and Hinglish (18.6%) but induced forgetting on general conversational greetings compared to UCIP-09 (16.2%). **UCIP-09 Medium Candidate remains the primary release artifact.**
