# CHAKRVIEW MODEL LIMITATIONS & FAILURE TAXONOMY
**Document Version:** `v0.10.0-alpha.1`  
**Target Model:** ChakrGenesis38M Conversational Alpha

---

## 1. Primary Scientific Grounding

ChakrView is trained **from scratch** using strict scientific isolation:
- **Zero External Pretrained Weights:** No LLaMA, Mistral, GPT, or BERT weights were used as initialization or teacher models.
- **Zero Hosted LLM APIs:** No OpenAI, Anthropic, or Google APIs were used to synthesize answers or label training data.
- **Training Budget:** Trained on ~4.17 million tokens of clean text across pretraining and SFT.

Because contemporary commercial conversational models are trained on $10^{11}$ to $10^{13}$ tokens (100,000x to 1,000,000x more data), **ChakrGenesis38M is an experimental small-model research artifact**, not a general-purpose artificial intelligence.

---

## 2. Failure Mode Taxonomy

### A. Semantic Drift & Grammatical Fragility
- **Symptom:** After generating a coherent opening clause (e.g. *"Hello! Ready to help..."*), continuation tokens often drift into tangential technical phrases or fragmented sentences.
- **Root Cause:** A 37.5M parameter network exposed to only 1 epoch of 4.17M tokens has acquired local bigram/trigram syntactic statistics and conversational templates, but lacks deep multi-layer world modeling.

### B. Factual Hallucination & Arithmetic Deficit
- **Symptom:** The model cannot reliably answer factual questions (e.g., boiling point of water, capitals, dates) or calculate basic arithmetic ($2 + 2$).
- **Root Cause:** Factual recall in transformers requires extensive parametric memorization across large corpus occurrences.

### C. Context Retention Decay
- **Symptom:** In multi-turn dialogue, the model fails to recall earlier stated entities (e.g., *"My favorite color is green"* -> *"What is my favorite color?"* yields unrelated words).
- **Root Cause:** Attention heads have not developed long-range positional binding under single-epoch pretraining.

### D. Multilingual Interleaving
- **Symptom:** When prompted in Devanagari Hindi or Hinglish, output may blend Devanagari scripts with English technical vocabulary in non-fluent arrangements.
- **Root Cause:** Multilingual token density in Corpus V6 (14.4% Hindi) provides sufficient structure for tokenizer compression and vocabulary acquisition, but requires further multi-epoch exposure to solidify separate language state spaces.

---

## 3. Explicit Attribution & Safety Notice

1. **Pure Neural Attribution:** When running `chat_cli.py`, all generated tokens are the direct probabilistic output of `ChakrGenesis38M`. No deterministic rule engine, hardcoded template, or external search API is masking the neural output.
2. **Safety Notice:** Do not rely on this model for advice, decision-making, code execution, or information retrieval.
