# How to Run ChakrView v0.9.1 Conversational Model

## Requirements
- Python 3.10+
- PyTorch (CPU or CUDA)
- SentencePiece

## Quickstart

### 1. Launch Interactive Chat CLI
```bash
python -m chakrview chat
```
Available in-session commands:
- `/info`: Inspect active checkpoint SHA-256, parameter count, and decoding settings.
- `/reset`: Clear chat memory and begin a fresh conversation turn.
- `/exit`: Terminate chat session.

### 2. Run Automated Verification Suite
```bash
python -m chakrview chat --test-mode
```

### 3. Programmatic Python API
```python
import torch
from pathlib import Path
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

ckpt_path = Path("releases/v0.9.1_conversational_preview/chakrview_v091_model.pt")
runner = DirectNeuralRunner(ckpt_path)
tok = runner.tokenizer

cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)

prompt = "User: Namaste! Aap kaise hain?\nAssistant:"
tokens = tok.encode(prompt, add_bos=True, add_eos=False)
curr = list(tokens)
gen = []

with torch.no_grad():
    for step in range(48):
        t_in = torch.tensor([curr], dtype=torch.long)
        logits = runner.model(t_in)[0, -1, :]
        tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
        gen.append(tok_id)
        curr.append(tok_id)
        if tok_id == 1:
            break

response = tok.decode(gen, errors="replace").strip()
print("ChakrView Response:", response)
```
