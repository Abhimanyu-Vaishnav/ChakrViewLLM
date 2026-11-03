# ChakrView v0.9.0 — Installation and Quickstart Guide

## 1. System Requirements
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **CPU:** Any modern x86_64 or ARM64 processor (Intel, AMD, Apple Silicon)
- **RAM:** Minimum 4 GB RAM (8 GB recommended)
- **Disk Space:** ~500 MB for repository and canonical model checkpoints
- **Python:** Python 3.10 to 3.14

---

## 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
   cd ChakrViewLLM
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Quickstart Inference Example

Run local interactive inference directly on CPU:

```python
from pathlib import Path
from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter, CANONICAL_CHECKPOINTS_ALLOWLIST

ckpt_path, expected_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST["step_10000"]
adapter = NeuralRuntimeAdapter(checkpoint_path=ckpt_path, max_new_tokens=32, decoding="greedy")

prompt = "ChakrView is an indigenous neural language model"
result = adapter.generate(prompt)

print(f"Generated text: {result['generated_text']}")
print(f"Tokens: {result['output_token_count']}, Latency: {result['latency_ms']:.1f}ms")
```
