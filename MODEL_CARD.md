---
language:
- en
- hi
tags:
- chakrview
- indigenous-ai
- text-generation
- custom-architecture
license: other
inference: false
---

# MODEL CARD: ChakrGenesis38M Conversational Alpha (`v0.10.0-alpha.1`)

## 1. Model Details & Compatibility Classification
- **Model Name:** ChakrGenesis38M-Conversational-Alpha
- **Version:** `v0.10.0-alpha.1`
- **Release Date:** November 2026
- **Architecture:** Autoregressive Decoder-only Transformer (`ChakrGenesis38M`)
- **Compatibility Classification:** **Custom PyTorch Standalone Package**
  > [!IMPORTANT]
  > **Inference Interface:** This model uses ChakrView's custom architecture and byte-level BPE tokenizer. It is **NOT** a native Hugging Face `transformers` model (`AutoModelForCausalLM` / `AutoTokenizer` are unsupported). Loading and inference must be performed using the packaged `modeling_chakrgenesis.py` and `tokenization_chakrview.py` modules.
- **License:** **Unspecified** (Pending final source licensing review).
- **Primary Weights Artifact:** `model.safetensors` (143.06 MB, SHA-256: `88c7ecd817648356a3e399872682c20a59bebc02873f7af6ae7e795cf8f66724`).
- **Original SFT Checkpoint SHA-256:** `e4beba3ae20f66eccc2cd06a45582087a059eb9117ed15b8ed9ab8e5500ff106`.

---

## 2. Architectural Specifications
- **Total Unique Parameters:** 37,499,392 (37.5M parameters)
- **Layers:** 12
- **Query Heads:** 8
- **Key/Value Heads:** 4 (Grouped Query Attention)
- **Hidden Dimension ($d_{\text{model}}$):** 512
- **FFN Hidden Dimension:** 1,408 (SwiGLU)
- **Vocabulary Size:** 4,096 (Byte-Level BPE)
- **Positional Encoding:** RoPE (Rotary Position Embeddings)
- **Context Window:** 256 tokens (configurable up to 512)
- **Weight Tying:** Active (`embedding.weight` is tied to `lm_head.weight`)

---

## 3. Training & Pretraining Provenance
- **From-Scratch Pretraining (Corpus V6):**
  - Exposed to 4,170,667 active training tokens across 9,843 documents (Python code/docs, STEM prose, Devanagari Hindi, and curated dialogue).
  - 1 epoch (1,305 optimizer steps), AdamW optimizer, cosine decay ($2.5 \times 10^{-4} \to 2.5 \times 10^{-5}$).
  - Pretrained base validation loss: 2.9562 (PPL: 19.22, Accuracy: 41.84%).
- **Conversational Supervised Fine-Tuning (SFT):**
  - Mixture of 1,612 uncontaminated dialogues (English 90%, Hindi 7%, Hinglish 3%).
  - Contamination Firewall: Filtered 75 items against 26,590 protected 8-grams from Benchmark V3, V2, and held-out test splits.
  - Response-only loss masking (`ignore_index=-100` on prompt tokens).
  - SFT duration: 3 epochs (273 steps), learning rate $5 \times 10^{-5}$.
  - SFT validation loss: **2.8924** (Next-token response accuracy: **49.70%**).

---

## 4. Standalone Loading & Inference Walkthrough

To run inference in any Python environment with only `torch` and `safetensors`:

```python
import torch
import safetensors.torch
from modeling_chakrgenesis import ChakrGenesis38M, Genesis38MConfig
from tokenization_chakrview import load_tokenizer_artifacts

# 1. Load configuration and model
config = Genesis38MConfig()
model = ChakrGenesis38M(config)
safetensors.torch.load_model(model, "model.safetensors", strict=True)
model.eval()

# 2. Load tokenizer
tokenizer, _ = load_tokenizer_artifacts(".")

# 3. Encode prompt and generate
prompt = "User: Hello! Who are you?\nAssistant:"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], dtype=torch.long)

generated = []
with torch.no_grad():
    for _ in range(32):
        logits, _ = model(input_ids)
        next_token = torch.argmax(logits[0, -1, :]).item()
        if next_token in [0, 2]: # PAD or EOS
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

print("Generated:", tokenizer.decode(generated))
```

---

## 5. Explicit Limitations & Failure Modes
This model is an **experimental research artifact** trained on ~4.17M tokens:
1. **Semantic Drift:** Extended completions drift into tangential phrasing.
2. **Factual Hallucination:** Fails exact factual queries; cannot calculate arithmetic reliably.
3. **Context Decay:** Multi-turn conversational memory decays over turns.
4. **Unsupported Use:** Not suitable for production advisory, medical, legal, or commercial assistant deployment.
