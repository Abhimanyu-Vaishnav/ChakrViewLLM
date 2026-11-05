# CR-001: ARCHITECTURAL DECISION RECORDS (ADR)

This document formalizes the fundamental design choices, rejected alternatives, and scientific justifications established in the CR-001 Cognitive Architecture.

---

## ADR-01: Preserving ChakrMicro v0.1 as an Untouchable Canonical Baseline
- **Context**: The existing ChakrMicro v0.1 (3.44M parameters) represents the foundational indigenous research anchor. Modifying its layer count, hidden dimension, or vocabulary would invalidate historical lineage and checkpoints.
- **Decision**: ChakrMicro v0.1 remains frozen. All cognitive scaling and architectural experiments must be developed in isolated experimental branches with separate configurations.
- **Consequences**: Ensures 100% reproducibility and scientific control.

---

## ADR-02: Rejection of Textual Chain-of-Thought as Sufficient Proof of Reasoning
- **Context**: Large language models often generate words like *"Therefore,"* or *"Step 1:"* through statistical co-occurrence without internal verification of mathematical or logical validity.
- **Decision**: ChakrView adopts a 4-way evaluation metric ($A_{final} \land V_{steps} \land C_{step\text{-}ans}$). Text that looks like reasoning but contains invalid intermediate steps or fails on random operand perturbations is graded as 0.
- **Consequences**: Prevents superficial chatbot imitation; forces engineering toward genuine inductive circuits.

---

## ADR-03: Curriculum-First Approach over Premature Architectural Complexity
- **Context**: It is tempting to add external neural memory networks, recurrent loops, or mixture-of-experts (MoE) to small models to force reasoning.
- **Decision**: Prioritize Level 1–3 curriculum data engineering on the standard transformer backbone before introducing architectural complexity.
- **Consequences**: We first measure what a standard 6-layer transformer can learn when trained with clean, dense variable-binding signals before spending compute on architectural variants.

---

## ADR-04: Strict Separation of Internal Cognition and External Augmentations
- **Context**: In AI literature, models often report high reasoning benchmarks by relying heavily on Python tool execution, external calculators, or vector retrieval.
- **Decision**: External RAG, calculators, and search are classified as tools, not internal neural capabilities. All benchmark reports must explicitly distinguish between **Neural-Only** mode and **Tool-Augmented** mode.
- **Consequences**: Upholds epistemic honesty and transparency.

---

## ADR-05: Hardware-Adaptive, CPU-First Parameter Boundary
- **Context**: ChakrView is designed for indigenous sovereign deployment on accessible, commodity hardware without requiring expensive GPU clusters.
- **Decision**: Cap experimental research architectures at $\le 20\text{M parameters}$ for CPU pretraining, maintaining training RAM under 250 MB and inference latency under 5 ms per token.
- **Consequences**: Guarantees anyone with a standard consumer laptop can train, evaluate, and deploy ChakrView.

