# ChakrView

ChakrView is an indigenous AI research initiative building modular neural and cognitive systems from first principles. It focuses on hardware-efficient, low-resource architectures designed to execute natively on CPUs and commodity systems without relying on third-party pretrained weights.

## What ChakrView Is

ChakrView explores how structured cognitive and relational mechanisms can be layered around a compact neural language core:
- **Indigenous Architecture**: Built from mathematical foundations and native code, not fine-tuned on external models.
- **CPU-First Execution**: Designed and tested natively on commodity CPU hardware without requiring GPUs or CUDA.
- **Low-Resource Neural Foundation**: Uses **ChakrMicro** (a 3.44M-parameter autoregressive transformer) as a compact representation layer.
- **Modular Cognition**: Separates the neural baseline from higher-order associative, relational, and reasoning modules.
- **Empirical Research System**: Governed by strict machine-verifiable capability contracts, reproducibility audits, and adversarial anti-shortcut benchmarks.

## Current Release

### ChakrView v0.1 — Verified Cognitive Core & Relational Acquisition Prototype

ChakrView v0.1 represents a verified research milestone: the successful acquisition and stabilization of **Intelligence Milestone I4 (Two-Hop Compositional Relational Binding)** under a controlled, multi-seed benchmark.

This release isolates a 69.8K-parameter **Neural Relational Acquisition Module** operating alongside the immutable ChakrMicro baseline.

## What Is Actually Verified

| Capability | Status | Evidence |
|:---|:---|:---|
| **CPU-First Runtime** | Verified | Native execution on CPU (`PyTorch CPU`), <290 MB RSS memory footprint |
| **Baseline Immutability** | Verified | Bit-exact ChakrMicro weights (`3,443,136` parameters, SHA-256 `c5571c...00a282da`) |
| **Tokenizer Integrity** | Verified | Exact SHA-256 match on BPE tokenizer artifacts (vocab size: 4,096) |
| **Candidate Isolation** | Verified | Relational module runs independently with zero baseline weight mutation ($\Delta W = 0$) |
| **I4 Compositional Binding** | Verified | Multi-seed G4 generalization mean = **77.78%** (min seed = 66.67%) on disjoint token pools |
| **Relational Key Routing** | Verified | Hop-1 routing mean = **100.00%**, Hop-2 routing mean = **88.89%** |
| **Language Retention** | Verified | Language modeling retention score $\ge 0.9500$ across training curricula |
| **Contamination Control** | Verified | Bit-exact zero token overlap between training and evaluation splits |
| **Anti-Shortcut Resilience** | Verified | Passed adversarial candidate, query, premise, and distractor permutations |

## What Is NOT Yet Verified

ChakrView v0.1 is an early research foundation. The following are **explicitly unsupported and unverified**:
- **Artificial General Intelligence (AGI)** or human-level reasoning.
- **Broad, open-domain reasoning** across arbitrary world knowledge.
- **Conversational instruction following or consumer chatbot abilities**.
- **Robust long-horizon continual learning** without task interference.
- **Autonomous self-improvement** in the general sense.
- **Three-hop or deeper compositional reasoning** (Milestone I5 remains in active research).
- **Production-scale distributed cluster intelligence**.
- **Universal hardware acceleration across arbitrary heterogeneous chips**.

## Important Generation Limitation

> [!WARNING]
> **Open-ended conversational text generation is NOT a claimed capability of ChakrView v0.1.**

The default runtime (`LocalModelRuntime.from_default()`) loads the canonical **3.44M-parameter ChakrMicro baseline**. Because this is a microscopic prototype trained on foundational pretraining text and has not undergone large-scale instruction tuning or chat alignment:
1. Under default greedy decoding (`temperature = 0.0`), unconstrained conversational prompts (such as `"Hello"`, `"What is ChakrView?"`, or `"India is a country."`) collapse into autoregressive suffix loops (e.g., repeating the trailing token until `MAX_TOKENS`).
2. Stochastic sampling (`temperature = 0.7, top_p = 0.9, repetition_penalty = 1.2`) eliminates repetitive loops and outputs diverse corpus fragments, but does **not** turn the microscopic baseline into a coherent conversational assistant.
3. The verified I4 compositional reasoning mechanism operates through structured relation episodes, **not** through open-ended generative chat.

## Architecture

The system distinguishes between representation, cognitive modules, and execution:

```
ChakrMicro Baseline (3.44M params, Autoregressive Transformer)
      ↓
Cognitive / Runtime Interfaces (Embedding projections, KV-cache)
      ↓
Specialized Cognitive Modules (e.g., NeuralRelationalAcquisitionModule, 69.8K params)
      ↓
Benchmark & Capability Governance (I4 Contract, Model Registry, Verification Gates)
```

The relational module does **not** replace the neural baseline; it functions as a modular cognitive layer that learns relational role projections, relative value kernels, and recurrent state transitions.

## I4 Benchmark Evidence

Evaluation across strict random seeds ($42, 101, 2026$) on the I4 Compositional Binding Benchmark:
- **G4 Compositional Generalization Mean**: **77.78%** (Seed 42: 66.67%, Seed 101: 66.67%, Seed 2026: 83.33%).
- **Minimum Seed G4**: **66.67%** (Exceeds the 40.0% safety floor).
- **Hop-1 Key Routing**: **100.00%** mean across seeds.
- **Hop-2 Key Routing**: **88.89%** mean across seeds.
- **Language Retention**: **0.9500** floor maintained.
- **Contamination**: Bit-exact zero token leakage.
- **Decision Gate**: `I4_ACHIEVED` (Candidate frozen in manifest).

*Note: In the benchmark, G4 measures target token identification accuracy over disjoint unseen token pools; Hop-1 and Hop-2 routing measure attention alignment with the ground-truth relational keys.*

## Quick Start

### 1. Prerequisites
- Python 3.10 to 3.14 (64-bit)
- Git

### 2. Setup
```powershell
# Clone and enter directory
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU dependencies
pip install -r requirements.txt torch
```

### 3. One-Command Release Verification
Run the unified verification script to audit all 7 release dimensions:
```bash
python scripts/verify_release.py
```

### 4. Running the Official Release Benchmark
```python
from chakrview.cognition.release_benchmark import run_release_benchmark

report = run_release_benchmark()
print(report.to_json())
```

### 5. Running the Research Prototype Runtime
```python
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.pipeline import GenerationConfig

# Initialize CPU runtime
rt = LocalModelRuntime.from_default()

# Note: Research prototype generation; not a conversational capability
cfg = GenerationConfig(max_new_tokens=16, sampling=SamplingConfig(temperature=0.7, top_p=0.9))
res = rt.generate("ChakrView", generation_config=cfg)
print("Generated text:", res.text)
```

## Repository Structure

```text
ChakrView/
├── chakrview/
│   ├── brain/              # ChakrMicro autoregressive neural core
│   ├── runtime/            # CPU-first local runtime, pipeline, sampling
│   ├── cognition/          # Relational acquisition module, benchmarks, manifests
│   ├── memory/             # Working and structured memory abstractions
│   ├── capability/         # Machine-verifiable capability contracts
│   ├── tokenizer/          # BPE tokenizer engine and serialization
│   └── release_manifest.py # Authoritative v0.1 release manifest
├── data/experiments/       # Tokenizer vocabulary and merge tables
├── docs/                   # Detailed research wave reports and audits
├── scripts/
│   └── verify_release.py   # One-command release verification runner
└── tests/                  # 180+ unit, integration, and regression tests
```

## Research Roadmap

### Completed & Verified (v0.1)
- **I1**: Next-Token Prediction on foundational text.
- **I2**: In-Context Associative Retrieval ($\ge 90\%$).
- **I3**: Contextual Retrieval with Distractors ($\ge 50\%$).
- **I4**: Two-Hop Compositional Relational Binding (**77.78%** G4 mean).
- Machine-verifiable capability contract and model registry lifecycle.

### Active & Future Research
- **I5 Investigation**: Three-hop compositional reasoning.
- **Relational Stabilization**: Eliminating residual ~11% Hop-2 routing variance.
- **Long-Horizon Retention**: Continuous learning without degradation of baseline language modeling.
- **Hardware Adaptation**: Dynamic computation budgeting across constrained devices.
- **Cognitive Self-Healing & Governance**: Audited safe candidate updates.

## Limitations

1. **Generation Repetition**: Unconstrained autoregressive generation on open prompts exhibits suffix loops under greedy sampling.
2. **Context Horizon**: Relational module sequence length is currently bounded at 128 tokens.
3. **Seed Variance**: While all seeds pass the I4 gate, variance exists ($66.7\%$ to $83.3\%$).
4. **Hardware Validation**: Tested CPU-first; multi-GPU distributed paths are not validated in v0.1.

## License

License: pending final project decision.

## Release Status

ChakrView v0.1 release candidate prepared (`PUBLIC_RELEASE_READY_FOR_FOUNDER_APPROVAL`). Public publication requires final founder acceptance.
