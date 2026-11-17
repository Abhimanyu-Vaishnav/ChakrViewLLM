# ChakrView UCIP-08 Medium Candidate Model Card

## Model Details
- **Model Name:** ChakrView Medium (UCIP-08 Candidate)
- **Architecture Family:** Indigenous Causal Transformer Decoder
- **Parameter Count:** 12,932,736
- **Layer Count:** 7
- **Model Width (Hidden Dim):** 384
- **Attention Heads:** 6
- **Feed-Forward Dimension:** 896
- **Vocabulary Size:** 8,192
- **Positional Encoding:** Rotary Positional Embeddings (RoPE)
- **Tied Embeddings:** Yes (token embedding weights tied with LM head projection)
- **Serialization File:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`
- **SHA-256 Digest:** `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631`
- **Parent Checkpoint:** `releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt` (`554c503e...`)

---

## Intended Use
- **Primary Domain:** CPU-first lightweight local conversational assistant for edge devices.
- **Language Support:** English, Hindi (Devanagari), Romanized Hindi, Hinglish.
- **Out of Scope:** Autonomous decision making, safety-critical medical/legal guidance, high-precision arithmetic computation without tool assistance.

---

## Evaluation Standard & Benchmark Performance
- **Evaluation Suite:** Sealed Owner Reality 100-Probe Suite (`sealed_owner_reality_100.json`, SHA-256: `7607eaf4...`)
- **Overall Benchmark Accuracy:** **10.0%** (10 / 100 probes passed)
- **Domain Performance Breakdown:**
  - English Conversation: 13.3% (2/15)
  - English Explanations: 0.0% (0/10)
  - Devanagari Hindi: 26.7% (4/15)
  - Roman Hindi: 10.0% (1/10)
  - Hinglish: 13.3% (2/15)
  - Exact Copying: 0.0% (0/10)
  - Elementary Arithmetic: 10.0% (1/10)
  - Instruction Following: 0.0% (0/10)
  - Semantic Consistency: 0.0% (0/5)

---

## Known Limitations
1. **Model Capacity Constraints:** At 12.93M parameters, general factual world knowledge and multi-step symbolic deduction (arithmetic, strict grammar-following) cannot be reliably internalized in parameter weights alone.
2. **Decoding Sensitivity:** Applying aggressive repetition penalties (>1.10) induces lexical divergence onto rare tail tokens.
3. **Factual Hallucination:** Explanations of complex scientific mechanisms may associate partially relevant keywords in broken syntax.
