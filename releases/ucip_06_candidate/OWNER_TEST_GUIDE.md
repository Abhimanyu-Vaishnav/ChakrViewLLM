# UCIP-06 Owner Testing & Verification Guide

This guide enables the owner to reproduce evaluation results and directly test the candidate models using the ChakrView CLI or interactive Python session.

---

## 1. Candidate Checkpoints

The candidate checkpoints have been staged in `releases/ucip_06_candidate/`:

- **ChakrMedium (12.93M parameters, 7 layers, 384 dim, 6 heads, 896 ffn):**
  - Path: `releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt`
  - SHA-256: `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72`

- **ChakrMicro (3.44M parameters, 6 layers, 192 dim, 6 heads, 512 ffn):**
  - Path: `releases/ucip_06_candidate/chakrview_ucip06_micro_model.pt`
  - SHA-256: `7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71`

---

## 2. Testing via CLI

You can chat directly with the candidate model:

```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt --memory off
```

---

## 3. Running Diagnostic Probes in Python

To run the exact owner evaluation set programmatically:

```powershell
.\.venv\Scripts\python.exe -c "
import torch
from pathlib import Path
from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

device = 'cpu'
ckpt_path = 'releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = ChakrMicro(ModelConfig(**ckpt['config']))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tokenizer, _ = load_tokenizer_artifacts(Path('data/experiments/vocab_4096'))

prompts = [
    'Hello ChakrView. How are you?',
    'What does a CPU do?',
    'Say exactly: The sky is blue.',
    'Aap kaise hain?',
    'नमस्ते! आप क्या कर सकते हैं?',
    'Bhai, machine learning ko simple language mein samjhao.',
    'If I have 5 apples and give 2 to my friend, how many apples remain?'
]

for p in prompts:
    formatted = f'User: {p}\nAssistant:'
    inp = torch.tensor([tokenizer.encode(formatted, add_bos=True, add_eos=False)], dtype=torch.long)
    curr = inp
    out = []
    with torch.no_grad():
        for _ in range(32):
            logits = model(curr)[0, -1, :]
            tok = torch.argmax(logits).item()
            if tok == 1: # EOS
                break
            out.append(tok)
            curr = torch.cat([curr, torch.tensor([[tok]])], dim=1)
    print(f'User: {p}')
    print(f'Assistant: {tokenizer.decode(out, errors=\"replace\").strip()}\n')
"
```

---

## 4. Re-Running the Sealed 500-Probe Benchmark

To independently re-verify the full 500-probe benchmark:

```powershell
.\.venv\Scripts\python.exe experiments/cognition/ucip_06/evaluator_500.py --checkpoint releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt
```
