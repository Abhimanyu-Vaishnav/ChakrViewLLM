# ChakrView Step 56: Modular Cognitive Architecture Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Architectural Specification
- **Subject**: Modular Cognitive Conditioning, External Memory Integration & Native Adapters

---

## 1. Foundational Architecture

The architecture rigorously decouples the immutable neural core from memory, external tools, environments, and execution controllers:

```
                 ┌──────────────────────────────────────┐
                 │       CHAKRVIEW NEURAL CORE          │
                 │      ChakrMicro v0.1 (3.44M)         │
                 │        ΔW_baseline ≡ 0               │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │      MODULAR COGNITIVE LAYER         │
                 │   (Native Low-Rank Task Adapters)    │
                 │   [Foundation]  [Reasoning]  [Code]  │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │    COGNITIVE CONTEXT CONDITIONER     │
                 │         <CORTEX_CONTEXT>             │
                 └──────────────────┬───────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
│    TASK MODE    │        │  MEMORY CONTEXT │        │ REASONING STATE │
│ [SYNTAX/REASON] │        │ (Episodic Store)│        │   (Diagnosis)   │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │         INFERENCE CONTROLLER         │
                 │     (Greedy / Top-P / KV-Cache)      │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │             CHAKRKSHETRA             │
                 │   Execute / Observe / Confinement    │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │          EPISODIC EXPERIENCE         │
                 │         (RIL Memory Record)          │
                 └──────────────────────────────────────┘
```

---

## 2. Invariant Decoupling Axioms

1. **Axiom 1: Core Decoupling**:
   $$\text{NEURAL CORE} \neq \text{MEMORY} \neq \text{TOOLS} \neq \text{CHAKRKSHETRA} \neq \text{EVALUATOR}$$
   The neural core must never be burdened with memorizing every ephemeral observation, API signature, or long-term episode.
2. **Axiom 2: Base Immutability**:
   The baseline weights (`c5571c...`) are bit-exact immutable ($\Delta W = 0$). Any specialization must reside either in **prompt conditioning** or in **independently serialized, detachable adapters**.
3. **Axiom 3: Reversibility**:
   Unloading an adapter or clearing context must restore the base neural core to its exact pristine behavior with zero residue.

---

## 3. Comparative Evaluation of Conditioning Approaches

| Approach | Parameter Overhead | CPU Latency Impact | Modularity / Reversibility | Anti-Forgetting Effectiveness | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Pure Prompt / Context Conditioning** | **0 parameters (0%)** | Negligible ($\sim 1\text{ ms}$) | High (Pure string formatting) | Moderate (Relies on model's existing attention) | **Adopted as Phase 4 Contract** |
| **B. Native Lightweight Low-Rank Adapter** | **$\sim 88\text{K}$ params ($\sim 2.5\%$)** | Very Low ($< 5\%$) | High (Detachable weights, isolated hash) | **High (Protects base parameters from mutation)** | **Adopted as Native Engine** |
| **C. External LoRA Framework (PEFT/HF)** | High dependency footprint | Moderate | Low (Introduces external libraries) | High | **Rejected (Violates Indigenous Principle)** |
| **D. Memory-Conditioned Prompting** | **0 parameters** | Negligible | High (Plugs into Episodic Memory) | High for dynamic facts and diagnostics | **Adopted as Phase 5 Bridge** |

### Selected Strategy: Dual Modular Conditioning
We implement a hybrid approach:
1. **Input-Level Conditioning**: A formal, token-bounded `<CORTEX_CONTEXT>` schema injecting relevant episodic memory, task mode, and diagnostic state.
2. **Weight-Level Conditioning**: A minimal, native PyTorch low-rank adapter (`NativeTaskAdapter`) applied to attention projections ($W_q, W_v$) that preserves the base model weights completely frozen.
