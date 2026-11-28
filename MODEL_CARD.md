# MODEL CARD: ChakrGenesis38M Conversational Alpha (`v0.10.0-alpha.1`)

## 1. Model Details
- **Model Name:** ChakrGenesis38M-Conversational-Alpha
- **Version:** `v0.10.0-alpha.1`
- **Release Date:** November 2026
- **Architecture:** Decoder-only Transformer (`ChakrGenesis38M`)
  - **Parameters:** 37,499,392 (37.5M parameters)
  - **Layers:** 12
  - **Query Heads:** 8
  - **Key/Value Heads:** 4 (Grouped Query Attention)
  - **Hidden Dimension ($d_{\text{model}}$):** 512
  - **FFN Hidden Dimension:** 1,408 (SwiGLU)
  - **Vocabulary Size:** 4,096 (Byte-Level BPE)
  - **Positional Encoding:** RoPE (Rotary Position Embeddings)
  - **Context Window:** 256 tokens
  - **Weight Tying:** Active (embedding layer shared with LM head)
- **Primary Checkpoint SHA-256:** `e4beba3ae20f66eccc2cd06a45582087a059eb9117ed15b8ed9ab8e5500ff106`
- **Pretrained Base Checkpoint SHA-256:** `92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`

---

## 2. Intended Use & Experimental Scope
- **Intended Use:** Pure neural baseline and research artifact exploring small-scale language acquisition from scratch under strict compute budgets without external model weights or hosted LLM API contamination.
- **Out of Scope / Unsupported Tasks:**
  - Production conversational assistant
  - Factual reference or search engine
  - Medical, legal, financial, or safety-critical guidance
  - Unsupervised code generation for production execution

---

## 3. Training Data & Pretraining Methodology
- **Pretraining Dataset:** `Corpus V6` (4,573,231 tokens total, 4,170,667 active training tokens across 9,843 documents).
  - Source Composition: Python standard library documentation/code (39.2%), STEM and Humanities encyclopedic prose (34.3%), Hindi Devanagari text (14.4%), Curated foundational dialogue/facts (10.9%), Sanskrit/English literature (1.2%).
  - Tokenizer: Canonical Byte-Level BPE 4096 (Vocab SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`).
- **Pretraining Schedule:** 1 complete epoch (1,305 optimizer steps), batch size 16, sequence length 256, AdamW optimizer ($\beta_1=0.9, \beta_2=0.98, \text{weight\_decay}=0.01$), cosine annealing from $2.5 \times 10^{-4}$ to $2.5 \times 10^{-5}$ with 100 warmup steps.
- **Supervised Fine-Tuning (SFT):**
  - Mixture of 1,612 uncontaminated dialogues (English 90.0%, Hindi 7.0%, Hinglish 3.0%).
  - Response-only loss masking: prompt tokens masked to `-100`, computing loss strictly on assistant response tokens.
  - SFT Duration: 3 epochs (273 steps), learning rate $5 \times 10^{-5}$.
  - SFT Validation Loss improved from 4.1777 (27.04% accuracy) to **2.8924** (49.70% accuracy).

---

## 4. Evaluation & Behavioral Findings

The model was subjected to a held-out qualitative and quantitative conversational evaluation across 12 distinct capability dimensions:
1. **UTF-8 Validity:** 100.0% valid token rendering.
2. **Repetition Rate:** Average 4-gram repetition rate of 0.15% (no degenerate infinite looping).
3. **Conversational Tone:** Capable of generating greetings and standard opening phrasing (e.g. *"Hello! Ready to help..."*).
4. **Code Syntax:** Generates syntactically recognizable Python function headers and docstring structures.
5. **Observed Failure Modes:**
   - **Semantic Drift:** Because the model has only seen ~4.17M tokens in pretraining (compared to hundreds of billions in standard LLMs), complex multi-turn reasoning and deep semantic coherence drift into word salad or thematic fragments.
   - **Factual Hallucination:** Does not possess reliable world knowledge or arithmetic ability.
   - **Language Interference:** Multilingual generation in Hindi occasionally mixes fragments from STEM contexts.

---

## 5. Hardware & Environmental Impact
- **Training Compute:** Trained entirely on local workstation hardware:
  - GPU: NVIDIA RTX A1000 6GB Laptop GPU
  - Pretraining Compute: 280.2 seconds (4.67 minutes)
  - SFT Compute: 54.4 seconds (0.91 minutes)
  - Peak VRAM Consumption: 2.057 GB (< 35% of GPU memory ceiling)
- **Inference Requirements:** Runs comfortably on both CPU and GPU with < 500 MB RAM allocation.
