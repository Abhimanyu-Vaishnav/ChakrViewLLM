# CV-GENESIS-NR-010 300-Prompt Benchmark Evaluation Report

## 1. Executive Summary

Mission **CV-GENESIS-NR-010 Phase 8** benchmarked the new `ChakrGenesis-17M` conversational model against the canonical 5.18M baseline (`genesis_sft_best.pt`) on **300 standardized blind evaluation prompts**.

Both models were evaluated under canonical decoding conditions:
- Temperature: 0.7
- Top-k: 20
- Max New Tokens: 40
- EOS Token ID: 1 (`<EOS>`)

---

## 2. Quantitative Results

| Model Architecture | Parameters | Coherent / Fluent Responses | Runaway Repetition Rate | Incomplete Rate |
| :--- | :--- | :--- | :--- | :--- |
| **ChakrGenesis-5M (NR-005 Best)** | 5,177,600 | 288 / 300 (**96.0%**) | **0.0% (0/300)** | 0.0% |
| **ChakrGenesis-17M (NR-010 SFT Best)**| **17,063,808** | 286 / 300 (**95.3%**) | **0.0% (0/300)** | 0.0% |

### Key Observations:
1. **Stability at 17M Scale:** The 17M model maintained a **95.3% coherence rate** with **0.0% repetition**, confirming that scaling up model depth and width did not destabilize RoPE attention or SwiGLU activations.
2. **Superior SFT Convergence:** The 17M model achieved a validation loss of **2.3930** (perplexity: **10.95**), outperforming the 5.18M best validation loss of 2.5314.
