# INFERENCE GUIDE: ChakrView Conversational Alpha (`v0.10.0-alpha.1`)

This guide explains how to run ChakrView locally on your CPU or GPU without external network access or hosted APIs.

---

## 1. System Requirements

- **Python:** 3.10, 3.11, 3.12, 3.13, or 3.14
- **PyTorch:** $\ge 2.0.0$ (CPU or CUDA-enabled GPU)
- **VRAM / RAM:**
  - GPU Execution: $\ge 1.5$ GB VRAM (RTX 3050, RTX A1000, GTX 1650 or higher)
  - CPU Execution: $\ge 2.0$ GB System RAM
- **Disk Space:** ~500 MB for repository and checkpoint artifacts.

---

## 2. Interactive Terminal Chat

Launch the interactive terminal chat session:

```powershell
# Using the project Python virtual environment
python chat_cli.py
```

### CLI Commands:
- `quit` or `exit`: Terminate the chat session.
- `reset`: Clear conversation history and reset context.

---

## 3. Python API Integration

You can easily integrate the generator into your own Python scripts:

```python
from chakrview.inference_package import ChakrViewGenerator, ChakrChatSession

# 1. Initialize generator (automatically detects CUDA if available)
generator = ChakrViewGenerator(
    checkpoint_path="experiments/cv24h_release/chakr_38m_conversational_sft.pt",
    tokenizer_path="data/experiments/vocab_4096"
)

# 2. Single-turn generation
prompt = "User: How do you write a function in Python?\nAssistant:"
response, metadata = generator.generate(
    prompt,
    max_new_tokens=64,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.15
)
print("Generated:", response)
print("Metadata:", metadata)

# 3. Multi-turn chat session
session = ChakrChatSession(generator)
turn1 = session.chat("Hello! What are you?")
print("Turn 1:", turn1)

turn2 = session.chat("Can you tell me more?")
print("Turn 2:", turn2)
```

---

## 4. Sampling Hyperparameters Guide

- **`temperature` (default: 0.7):** Controls sampling entropy. Set to `0.0` for deterministic greedy decoding. Lower values (0.2–0.5) produce more repetitive, structured tokens; higher values (0.8–1.0) introduce creative variation.
- **`top_k` (default: 40):** Limits candidate tokens at each step to the top $k$ highest probabilities.
- **`top_p` (default: 0.9):** Nucleus sampling cutoff threshold.
- **`repetition_penalty` (default: 1.15):** Penalizes tokens that already appear in the context to prevent degeneration and looping.
