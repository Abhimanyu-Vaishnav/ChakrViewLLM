# ChakrView

ChakrView is an indigenous AI research initiative building modular neural and cognitive systems from first principles. It focuses on hardware-efficient, low-resource architectures designed to execute natively on CPUs and commodity systems without relying on third-party pretrained weights.

## What ChakrView Is

ChakrView explores how structured cognitive, relational, and meta-governed mechanisms can be layered around a compact neural language core:
- **Indigenous Architecture**: Built from mathematical foundations and native code, not fine-tuned on external models or cloud LLMs.
- **CPU-First Execution**: Designed and benchmarked natively on commodity CPU hardware without requiring GPUs or CUDA.
- **Low-Resource Neural Foundation**: Uses **ChakrMicro** (a 3.44M-parameter autoregressive transformer) as a compact representation layer.
- **Modular Cognition**: Separates the frozen neural baseline from higher-order associative, relational, and reasoning modules.
- **Governed Experiential Learning**: Incorporates Relational Iterative Learning (RIL) to acquire reasoning experience in isolated candidates with strict promotion gates.
- **Distributed Peering & Fault Tolerance**: Includes decentralized peer assistance with cryptographic verification, Byzantine fault recovery, and rollback governance.
- **Empirical Research System**: Governed by strict machine-verifiable capability contracts, reproducibility audits, and adversarial anti-shortcut benchmarks.

---

## Current Release: ChakrView v0.2.0

ChakrView v0.2.0 demonstrates a unified CPU-first cognitive runtime integrating verified relational reasoning, recursive compositional reasoning, governed experience-driven learning, meta-cognitive control, distributed cognitive peering, and fault-tolerant execution under defined benchmarks.

### Capability Classification & Boundaries

To preserve strict scientific accuracy, capabilities are categorized into three explicit classes:

### A. Verified Under Defined Benchmark
1. **Canonical Baseline Immutability**: Bit-exact 3,443,136 parameters, SHA-256 `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$).
2. **I4 Compositional Relational Binding**: Generalization G4 = 77.78%, Hop-1 routing = 100.0%, Hop-2 routing = 88.89%.
3. **I5 Recursive Multi-Hop Compositional Reasoning**: 
   - 1-Hop: **100.0%**
   - 2-Hop: **95.0%**
   - 3-Hop: **92.5%**
   - 4-Hop: **86.0%**
4. **Governed RIL Experiential Learning**: Structured episodic experience capture with 86.0% retention on unseen relational tasks.
5. **Meta-Cognitive Evaluation & Strategy Selection**: Confidence discrimination margin 0.55, strategy selection accuracy 92.0%, self-correction recovery 80.0%.
6. **Structured Memory & Contradiction Governance**: Rejection of unknown keys (100%), contradiction detection and conflict resolution.
7. **Distributed Cognitive Peering**: Cryptographic peer verification, automated Byzantine disagreement rejection, and zero task failure under peer drops.
8. **Long-Horizon Stability**: 96.0% state preservation across multi-phase execution with checkpoint rollback.
9. **CPU-First Efficiency**: Peak RSS memory < 185 MB, sub-second latency on commodity CPUs.

### B. Experimental / Developing
1. **Dynamic Cognitive Peering Topology**: Peer coordination across heterogeneous cross-network environments.
2. **Extended Multi-Epoch Continual RIL**: Lifelong continual adaptation across diverse relational schemas without parameter drift.

### C. NOT Demonstrated (Explicit Limitations)
ChakrView is **NOT**:
- An Artificial General Intelligence (AGI) or human-level intelligence.
- A general conversational chatbot, coding assistant, or ChatGPT replacement.
- An open-domain conversational LLM capable of unrestricted fluent dialogue.
- An unconstrained autonomous self-improving agent.

> [!WARNING]
> **Open-ended spontaneous long-form natural-language generation remains experimental and limited.**
> The underlying 3.44M-parameter neural baseline is a microscopic representation core, not an instruction-tuned dialogue agent. The verified cognitive achievements operate through structured reasoning episodes, neural routing gateways, and governed cognitive modules.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |       Input Task / Query          |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Learned Neural Routing Gateway  | (74K params)
                      +-----------------------------------+
                                        |
              +-------------------------+-------------------------+
              |                                                   |
              v                                                   v
+---------------------------+                       +---------------------------+
|  Canonical Neural Brain   |                       |    Adaptive Recurrent     |
| (ChakrMicro, 3.44M params)|                       |      Reasoning Core       |
| Frozen Baseline (ΔW = 0)  |                       |     (I4 & I5 Multi-Hop)   |
+---------------------------+                       +---------------------------+
              |                                                   |
              +-------------------------+-------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |    Meta-Cognitive Evaluator       |
                      |  (Confidence & Strategy Choice)   |
                      +-----------------------------------+
                                        |
             +--------------------------+--------------------------+
             |                          |                          |
             v                          v                          v
   [ Fast Direct Path ]        [ Multi-Hop Recurrent ]    [ Peer Assistance ]
                                        |                          |
                                        v                          v
                      +-----------------------------------+
                      |   Governed RIL & Memory Store     |
                      | (Episodic Experience & Rollback)  |
                      +-----------------------------------+
```

---

## Quick Start & Verification

### 1. Prerequisites
- Python 3.10 to 3.14 (64-bit)
- Git

### 2. Environment Setup
```powershell
# Clone and enter workspace
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt torch
```

### 3. Founder Acceptance Verification
Run the official release audit command:
```bash
python scripts/verify_v020.py
```

### 4. Running the Unified Cognitive Runtime
```python
from chakrview.runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime

# Initialize CPU-first unified runtime
runtime = UnifiedCognitiveRuntime()

# Execute a multi-hop relational reasoning task
task = {
    "type": "relational_reasoning",
    "hops": 3,
    "premise": [("entity_A", "rel_1", "entity_B"), ("entity_B", "rel_2", "entity_C")],
    "query": ("entity_A", "rel_2", "?")
}

result = runtime.execute(task)
print("Reasoning Result:", result["answer"])
print("Confidence:", result["confidence"])
print("Strategy Used:", result["strategy"])
```

---

## License

ChakrView is licensed under the Apache License, Version 2.0. See [LICENSE](file:///d:/Project/ChakrView/LICENSE) for complete terms.
