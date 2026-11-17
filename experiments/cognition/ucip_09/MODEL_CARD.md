# ChakrView UCIP-09 Medium Candidate Model Card

## Model Details
- **Model Name:** ChakrView Medium (UCIP-09 Candidate)
- **Architecture Family:** Indigenous Causal Transformer Decoder
- **Parameter Count:** 12,932,736
- **Layer Count:** 7
- **Model Width ($d_{\text{model}}$):** 384
- **Attention Heads:** 6
- **Feed-Forward Dimension ($d_{\text{ff}}$):** 896
- **Vocabulary Size:** 4,096 byte-level BPE
- **Positional Encoding:** Rotary Positional Embeddings (RoPE, $\theta = 50,000.0$)
- **Normalization:** Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Tied Embeddings:** True
- **Serialization File:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`
- **SHA-256 Digest:** `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`
- **Parent Checkpoint:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt` (`fe6f381a...`)

---

## Intended Use
- **Primary Function:** CPU-first, lightweight local conversational assistant for edge workstations and embedded environments.
- **Languages:** English, Hindi (Devanagari), Romanized Hindi, Hinglish.
- **Out of Scope:** High-stakes medical/legal reasoning, zero-shot encyclopedic knowledge without hybrid retrieval.

---

## Benchmark Performance Matrix

### 1. Sealed Language Intelligence 500 Benchmark (`sealed_language_intelligence_500.json`)
- **Overall Accuracy:** **16.2%** (81 / 500 probes passed)
  - English conversation: 31.4% (22/70)
  - English explanations: 0.0% (0/70)
  - Devanagari Hindi: 38.6% (27/70)
  - Roman Hindi: 18.0% (9/50)
  - Hinglish: 17.1% (12/70)
  - Exact copying: 0.0% (0/40)
  - Elementary arithmetic: 7.5% (3/40)
  - Instruction following: 7.5% (3/40)
  - Semantic consistency: 20.0% (5/25)
  - Multi-turn context: 0.0% (0/25)

### 2. Frozen Owner Reality 100 Benchmark (`sealed_owner_reality_100.json`)
- **Overall Accuracy:** **10.0%** (10 / 100 probes passed)

---

## Known Boundaries & Next Steps
- **Conversational Tone:** Demonstrates fluent, complete opening identity generation (`"I am ChakrView, an independent neural conversational assistant for efficient local execution."`).
- **Parametric Weight Boundary:** Multi-concept scientific explanations require external memory integration (`chakrview/memory/`) rather than solely relying on 12.9M parameter weights.
