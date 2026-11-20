# ChakrView Conversation Alpha

Welcome to **ChakrView Conversation Alpha**, an indigenous lightweight neural conversational language model developed and fine-tuned from first principles.

---

## 1. Quickstart (100% Local CPU Execution)

No GPU, cloud connection, or external API keys are required. Runs completely locally on CPU with <500MB RAM.

### Run Dialogue Chat
```bash
python releases/chakrview_conversation_alpha/inference.py --prompt "User: Hello! Who are you?\nAssistant: "
```

### Run Smoke Test
Verify installation integrity and forward-pass execution:
```bash
python releases/chakrview_conversation_alpha/smoke_test.py
```

---

## 2. Model Specifications

- **Architecture:** ChakrMicro v0.1 (Decoder-only Causal Transformer, RoPE, RMSNorm, SwiGLU)
- **Parameters:** 3,443,136 unique parameters
- **Layers:** 6 | **Width ($d_{model}$):** 192 | **Heads:** 6 | **Hidden Dim:** 512
- **Context Ceiling:** 512 tokens
- **Vocabulary:** 4,096 tokens (BPE, byte-backed fallback, 724 Devanagari subword merges)
- **Training Lineage:** 
  - Foundational Pre-Training: UCIP-13 on Stage C corpus (6.55M tokens)
  - Supervised Fine-Tuning: UCIP-15 mixture tuning (80% SFT on 12k instruction records + 20% Stage C foundational replay)

---

## 3. Supported Capabilities & Boundaries

### Capabilities
- Conversational turn-taking with standard prompt wrappers (`User: ...\nAssistant: ...`).
- High English instruction compliance and contextual response framing.
- Multilingual script handling across English, Hindi (Devanagari), and Hinglish.
- Zero reliance on external pretrained teacher LLMs.

### Known Limitations
- At 3.44M parameters, open-domain factual knowledge and multi-digit arithmetic are limited.
- Unconstrained long-context memory beyond 512 tokens requires external state storage.
