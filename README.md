# ChakrView (चक्रव्यूह)

ChakrView is an indigenous AI research project being developed strictly incrementally from first principles.

## Vision

ChakrView is envisioned as an indigenous, hardware-efficient AI brain designed to execute efficiently on modern systems as well as older, low-resource computers and embedded/edge devices.

The long-term research roadmap encompasses:
1. **Indigenous Neural Language Core**: Designed from foundational principles without relying on third-party pretrained weights.
2. **Efficient Inference Runtime**: Optimized execution tailored to diverse and constrained target hardware.
3. **Memory System**: Contextual, persistent, and structured memory representations.
4. **Skill-Learning System**: Dynamic acquisition and consolidation of operational skills.
5. **Tool/Action System**: Extensible interfaces for environmental interaction and tool use.
6. **Adaptive Computation**: Dynamic allocation of compute based on task complexity.
7. **Reinforcement-Learning Based Skill Optimization**: Fine-tuning cognitive pathways and skills through environmental feedback.
8. **Hardware-Aware Execution**: Dynamic adaptation to heterogeneous compute resources (CPU, low-end GPU, memory constraints).

---

## Current Status (v0.1 Release Candidate)

ChakrView has achieved **Intelligence Milestone I4 (Two-Hop Compositional Binding)** under strict held-out evaluation across multiple random seeds, while maintaining the bit-exact canonical baseline.

### Verified Capabilities
- **Next-Token Prediction & Language Modeling (I1)**: Baseline ChakrMicro model.
- **In-Context Retrieval (I2)**: Key-value associative lookup.
- **Associative Contextual Retrieval (I3)**: Distractor resistance and generalization.
- **Compositional Relational Acquisition (I4)**: Multi-seed G4 generalization mean = 77.78%, Hop-1 key routing = 100%, Hop-2 key routing = 88.89%.
- **CPU-First Execution**: Fully functional without GPU or CUDA dependencies.
- **Safety & Baseline Immutability**: Canonical baseline parameters: `3,443,136`, SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.

### Explicitly Unsupported Claims (What v0.1 is NOT)
- **NOT Artificial General Intelligence (AGI)** or human-level intelligence.
- **NOT unrestricted open-domain reasoning** across arbitrary knowledge bases.
- **NOT autonomous self-improvement**.
- **NOT proven long-horizon continual learning** without forgetting.
- **NOT three-hop or arbitrary deep reasoning (I5 remains in research)**.

---

## Architecture & Model Summary

| Component | Architecture / Spec | Parameters | Integrity / SHA |
|:---|:---|:---|:---|
| **Canonical Baseline** | ChakrMicro (Autoregressive Transformer) | 3,443,136 | `c5571c9c...00a282da` |
| **Relational Core (I4)** | Neural Relational Acquisition Module | 69,809 | Manifest Verified |
| **Tokenizer** | Byte-Pair Encoding (BPE), Vocab = 4,096 | N/A | Hashes Verified |

---

## Quickstart

### Prerequisites
- Python 3.14 (or compatible 64-bit Python 3.10+)
- Git

### Environment Setup
```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install minimal dependencies
pip install -r requirements.txt torch
```

### Running the Local Runtime
```python
from chakrview.runtime.local_runtime import LocalModelRuntime

# Load default CPU-first runtime
rt = LocalModelRuntime.from_default()
result = rt.generate("ChakrView indigenous")
print(result.text)
```

### Running the Official Release Benchmark
```python
from chakrview.cognition.release_benchmark import run_release_benchmark

report = run_release_benchmark()
print(report.to_json())
```

---

## Known Limitations
1. **Seed Variance**: I4 scores exhibit variation across seeds (best: 83.33%, min: 66.67%).
2. **Context Window**: Current relational module evaluation is bounded at 128 tokens.
3. **Language Retention Floor**: Evaluated at 0.9500 threshold; unregularized tuning may cause drift.
4. **Hardware Validation**: Tested and verified CPU-first; distributed multi-GPU training pathways are unvalidated.

---

## Project Structure

```text
ChakrView/
├── chakrview/
│   ├── brain/          # Neural architectures and ChakrMicro baseline
│   ├── runtime/        # CPU-first local runtime, inference pipeline, integrity
│   ├── cognition/      # Relational modules, manifests, registries, benchmarks
│   ├── memory/         # Context and structured memory abstractions
│   ├── capability/     # Capability contracts and registries
│   └── tokenizer/      # BPE tokenizer engine and artifacts
├── data/experiments/   # Verified tokenizer vocab and merge files
├── docs/               # Research wave reports and forensic audits
└── tests/              # Full unit and regression test suite
```
