# ChakrView v0.9.1 Model Card (Conversational Preview)

## Model Details
- **Model Name:** ChakrView v0.9.1 (ChakrMicro v0.1)
- **Model Type:** Compact Autoregressive Decoder Transformer
- **Architecture Parameters:** 3,443,136 total parameters ($L=6, d=192, H=6, d_{\text{mlp}}=512, V=4096$)
- **Developer:** ChakrView Core Team (First Principles Indigenous Research)
- **Release Date:** November 11, 2026
- **Release Channel:** Experimental Preview (Research Release)

## Intended Use
- Interactive terminal chat on CPU-constrained edge hardware.
- Low-latency edge conversational assistance (greetings, identity, basic dialogue turns, Hinglish interaction).
- Research into capability emergence, representation bounds, and refusal mechanics in compact language models (<5M parameters).

## Performance & Evaluation
Evaluated on `CONVERSATION_BENCHMARK_V1.json` (200 diverse, frozen probes):
- **Refusal Sink Elimination:** Dropped from 26/200 (13.0%) in parent to **0/200 (0.0%)** in v0.9.1.
- **Self-Identity Recognition:** Correct identity and purpose statements in **66.7%** of direct identity probes (vs. 0.0% in parent).
- **Hinglish/Hindi Greeting Grounding:** Responds fluently to Hindi greetings (e.g. `User: Namaste! Aap kaise hain?` -> `Namaste! Main aapki kya madad kar sakta hoon?`).
- **Repetition Stability:** 100% non-repetition rate with calibrated decoding (`repetition_penalty = 1.15`, `no_repeat_ngram_size = 3`).

## Known Limitations & Boundaries
1. **Representational Capacity:** At 3.44 million parameters, the model cannot perform complex factual recall across unbounded encyclopedic domains or multi-step symbolic reasoning.
2. **Context Horizon:** Multi-turn conversational continuity is optimal for 1 to 2 turns. Contexts beyond 3 turns can suffer from entity confusion.
3. **Absence of Pretrained Backends:** Intelligence originates 100% from ChakrView's custom pretraining and supervised alignment. No external pretrained backends (e.g., Llama, GPT, Hugging Face Hub) are used.

## Usage
```bash
# Interactive terminal chat
python -m chakrview chat

# Automated test mode
python -m chakrview chat --test-mode

# Version info
python -m chakrview version
```
