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

## Public Release Candidate: ChakrView v0.5.0

ChakrView v0.5.0 is an indigenous neural-cognitive architecture integrating a CPU-first 3.44M-parameter neural core with controlled neural learning, relational reasoning, semantic grounding under defined tasks, memory, planning, evidence reasoning and governed adaptation.

Internal research lineage:
`V0.4 candidate -> V0.5 neural research -> V0.6 semantic research` (Candidate commit: `1e06746`).

### Capability Classification & Boundaries

To preserve strict scientific accuracy, capabilities are categorized into three explicit classes:

### A. Verified Under Defined Benchmark
1. **Canonical Baseline Immutability**: Bit-exact 3,443,136 parameters, SHA-256 `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$).
2. **Relational Semantic Grounding**: Normalizes varied natural-language surface phrasings into canonical relational triples `(Subject, Relation, Object)` with 75%+ invariance across surface variants.
3. **Multi-Hop Relational Transitivity**: Recursive reasoning over multi-hop chains ($E_1 \to E_2 \to E_3 \to E_4$) with transitivity score $\ge 0.65$ without lookup tables.
4. **Semantic Cognitive Coupling**: Seamless injection of relational semantics into cognitive decision gates, distinguishing factual answers, epistemic contradictions, and honest abstentions.
5. **Hierarchical Planning & Failure Recovery**: Goal decomposition into dependency-aware subgoals (`RETRIEVE`, `VERIFY`, `SYNTHESIZE`) with automatic replanning upon step failure.
6. **Task-Scoped Working Memory**: Bounded ephemeral working state (`WorkingMemory2`) with clean task-boundary resets and zero durable memory pollution.
7. **Evidence Precedence Over Stale Beliefs**: Newer verified external evidence supersedes outdated internal beliefs automatically.
8. **CPU-First Execution**: Fully native low-resource CPU execution without GPU or cloud dependency.

### B. Experimental / Developing
1. **Disjoint Entity Generalization**: Zero-shot relational reasoning across completely unseen arbitrary entity vocabularies.
2. **Multi-Dialect Grounding**: Rich Hindi/Hinglish deep semantic alignment beyond token and keyword cues.

### C. NOT Demonstrated (Explicit Limitations)
ChakrView is **NOT**:
- An Artificial General Intelligence (AGI) or human-level intelligence.
- A general conversational chatbot, coding assistant, or ChatGPT replacement.
- An open-domain conversational LLM capable of unrestricted fluent dialogue.
- A fully autonomous self-improving system without governance.
- A physical 28nm silicon tape-out (hardware targets are simulated and CPU-evaluated).

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
                      |   Language <-> Cognitive Bridge   |
                      |   (Intent & Frame Extraction)     |
                      +-----------------------------------+
                                        |
               +------------------------+------------------------+
               |                                                 |
               v                                                 v
+-------------------------------+               +-------------------------------+
|  Canonical Neural Brain       |               |  Relational Semantic Encoder  |
| (ChakrMicro, 3.44M params)    |               |  (Grounding & Transitivity)   |
| Frozen Baseline (ΔW = 0)      |               |  (Surface Invariance)         |
+-------------------------------+               +-------------------------------+
               |                                                 |
               +------------------------+------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |  Semantic Cognitive Coupling      |
                      |  (Evidence & Conflict Resolution) |
                      +-----------------------------------+
                                        |
              +-------------------------+-------------------------+
              |                                                   |
              v                                                   v
+-------------------------------+               +-------------------------------+
|   Hierarchical Planner        |               |  Bounded Working Memory 2.0   |
| (Goal Breakdown & Recovery)   |               |  (Task-Scoped, Zero Leaks)    |
+-------------------------------+               +-------------------------------+
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

### 3. Founder Manual Testing & Interactive Console
Launch the interactive founder console:
```bash
python scripts/founder_manual_test.py
```

Run the 15 predefined non-destructive founder test cases:
```bash
python scripts/founder_manual_test.py --suite
```

### 4. Release Verification
Execute full release candidate verification:
```bash
python scripts/verify_v050_release.py
```

### 5. Running the Unified Cognitive Runtime
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
