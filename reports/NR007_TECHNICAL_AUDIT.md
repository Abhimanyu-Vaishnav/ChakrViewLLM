# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-007
## COMPREHENSIVE TECHNICAL AUDIT & ARCHITECTURAL ASSESSMENT

**Mission ID:** CV-GENESIS-NR-007  
**Date:** 24 November 2026  
**Auditor:** Principal AI Research Engineer & ML Systems Architect  
**Subject:** ChakrGenesis Neural Engine & SFT Training/Inference Subsystems  

---

### 1. Neural Architecture Audit: `ChakrGenesis`

ChakrGenesis is an independently implemented decoder-only Transformer with modern architectural primitives:
- **Parameter Count:** **5,177,600 parameters** (~5.18M).
- **Hidden Dimension ($d_{model}$):** 256.
- **Layers ($n_{layers}$):** 6 transformer blocks.
- **Attention Configuration:** Grouped-Query Attention (GQA).
  - Query heads ($n_{q\_heads}$): 8 (head dimension: 32).
  - Key/Value heads ($n_{kv\_heads}$): 2 (head dimension: 32).
  - KV-cache reduction factor: $4\times$.
- **Positional Encoding:** Rotary Position Embedding (RoPE) applied to query and key states before dot-product attention.
- **Feed-Forward Layer:** SwiGLU activation ($\text{Swish}(x W_{gate}) \cdot (x W_{up})$ followed by $W_{down}$) with intermediate dimension 682.
- **Normalization:** RMSNorm applied pre-attention and pre-FFN, plus a final RMSNorm layer.
- **Output Layer:** Weight-tied output projection (`embedding.weight` reused as the language model head), conserving ~1.05M parameters.
- **Sequence Context:** 512 tokens maximum training/inference window.

#### Architectural Evaluation & Capacity Ceiling:
At 5.18M parameters, the neural model operates in the sub-10M parameter regime.
- **Strengths:** Extremely high throughput (~31,800 tokens/sec on RTX A1000 GPU), minimal memory footprint (2.42 GB peak VRAM), fast experimentation, and instant cold-start loading (<0.5 seconds).
- **Limitations:** Limited parametric capacity for broad world-knowledge memorization and deep multi-turn variable tracking. It functions exceptionally as a linguistic routing and dialogue core, but requires external cognitive augmentation for long-horizon memory and exact multi-digit math.

---

### 2. Tokenizer Architecture Audit

- **Type:** Byte-level Byte-Pair Encoding (BPE).
- **Artifact:** `data/experiments/vocab_4096/vocab.json` (SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`).
- **Vocabulary Size:** 4,096 tokens.
- **Special Tokens:** `<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2.
- **Strengths:** 100% byte-level coverage prevents out-of-vocabulary crashes. Absorbs typos and handles English, Devanagari Hindi, and Python syntax without fallback errors.
- **Audit Findings:** The previous token permutation flaw discovered in NR-003 was completely resolved in NR-004. The tokenizer is verified to match the Stage-C corpus tokens and SFT checkpoints.

---

### 3. SFT Training Pipeline & Masking Audit

- **Loss Masking:** Masked cross-entropy applies `-100` to user prompt tokens, ensuring gradients flow solely through assistant responses.
- **EOS Supervision:** `<EOS>` (ID 1) appended and supervised at the end of each assistant turn. This successfully brought runaway repetition down from 58.4% to **0.0%**.
- **Optimization:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.01), cosine learning rate schedule with linear warmup, and automatic mixed precision (AMP).
- **Generalization:** Monotonic validation loss reduction from 4.6458 to 2.4546 (PPL 11.64).

---

### 4. Technical Debt & Strategic Improvement Priorities

1. **Multilingual Dataset Asymmetry:**
   - In NR-005, the training mixture was 75.6% English, 12.6% Hinglish, and 11.8% Hindi.
   - NR-006 proved this asymmetry directly causes Hindi factual precision to lag (3.0% strict vs. 18.1% in English).
   - *Action:* The next iteration of dataset engineering must balance Hindi and Hinglish to $\ge 30\%$ each.
2. **Cognitive Hybrid Separation:**
   - Single-turn entity recall works (60%), but multi-turn state updates fail.
   - *Action:* Keep the neural model strictly focused on language fluency, syntactic completion, and conversational personality, while explicitly coupling it with an **Episodic Working Memory Buffer** and **Span Pointer Network** for multi-turn factual state retention.
