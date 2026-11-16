# UCIP-07 Model Card: ChakrMedium Candidate

## 1. Model Details

- **Model Name**: ChakrView Medium UCIP-07 Candidate
- **Version**: 0.7.0-candidate
- **Parameter Count**: 12,932,736
- **Architecture**: Decoder-only Autoregressive Transformer
- **Layers**: 7
- **Model Width ($d_{model}$)**: 384
- **Attention Heads**: 6 ($d_{head}=64$)
- **FFN Hidden Dimension**: 896 (SwiGLU)
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)
- **Normalization**: Pre-RMSNorm ($\epsilon=10^{-5}$)
- **Vocabulary**: 4,096 BPE tokens (tied embeddings)
- **Context Length**: 512 tokens
- **Artifact SHA-256**: `554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2`

---

## 2. Intended Use

- Local, CPU-first conversational inference.
- Multi-turn instruction following, basic QA, and bilingual English / Hindi / Hinglish dialogue.
- Educational experimentation and research on small-scale indigenous neural language models.

---

## 3. Evaluation Performance

Evaluated against the frozen 800-probe sealed benchmark (`sealed_800_benchmark.json`):

- **Overall Accuracy**: **31.37% (251 / 800)**
  - English Conversation: **41.00%**
  - English Sentence Generation: **27.14%**
  - English Explanations: **15.00%**
  - Devanagari Hindi: **42.00%**
  - Roman Hindi: **58.75%**
  - Hinglish: **55.00%**
  - Elementary Arithmetic: **10.00%**
  - Instruction Following: **31.67%**
  - Semantic Consistency: **14.29%**
  - Exact Copying: **0.00%**

---

## 4. Limitations & Known Issues

1. **Exact Copying Failure**: Does not strictly mirror input without adding conversational commentary.
2. **Multi-Step Arithmetic**: Struggles with arithmetic operations beyond single-digit additions and subtractions.
3. **Attractor Trapping Under Divergent Prompts**: If exposed to wild inputs outside training distributions, the model may drift towards familiar conversational structures.
