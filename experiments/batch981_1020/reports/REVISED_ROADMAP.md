# WP 1015–1017: Revised Evidence-Based Strategic Roadmap

## 1. Principles of the Strategic Reset

1. **Stop Building Heuristic Scaffolding Around Synthetic DSLs**:
   Halt all further attempts to force the 3.8M core model to generate multi-step Plan DSL syntax. The DSL has become an impedance mismatch.
2. **Prioritize Practical Utility Over Abstract Formalisms**:
   Directly connect user inputs to tools using proven, simple representations (JSON slots or direct tool calls).
3. **Re-anchor on Core Neural Capability Expansion**:
   Train the core model on diverse multi-task data with experience replay to prevent catastrophic forgetting.

---

## 2. Three Parallel Strategic Tracks

```
                        ChakrView Master Roadmap
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  [Track A: Core Neural]   [Track B: Practical Utility]  [Track C: Systems Base]
  - General pretraining     - Local Document Q&A App     - Quantization (FP16/INT8)
  - Balanced multi-task     - Interactive Terminal CLI   - Automated benchmark CI
  - Replay memory buffer    - Verified Memory Assistant  - Distributed validation
```

---

## 3. Next Five Milestones (Ranked by Strategic Value)

### Milestone 1: Local Knowledge Assistant MVP (Batches 1021–1060)
- **Objective**: Deliver a usable local application that answers questions from user-provided Markdown/PDF documents with verified source citations and multi-turn conversational memory.
- **Why #1**: Delivers tangible user value immediately using ChakrView's strongest verified assets (retrieval, memory, evidence gate) without relying on fragile multi-step neural planning.
- **Pass Criterion**: >85% citation accuracy on 50 held-out local document queries.

### Milestone 2: Pretraining Scale-Up & Replay Curriculum (Batches 1061–1100)
- **Objective**: Scale pretraining from step 10,000 to step 25,000 using mixed multi-task data (language, basic arithmetic, Q&A, and simple tool JSON) with a 20% replay buffer to prevent forgetting.
- **Pass Criterion**: Model maintains general language loss < 2.0 while achieving >70% on held-out factual reasoning.

### Milestone 3: Direct JSON Tool Calling with Numeric BPE Normalization (Batches 1101–1140)
- **Objective**: Replace Plan DSL with standardized compact JSON tool calls (`{"tool": "calculator", "expr": "600*5"}`) and implement digit-level token grouping to permanently eliminate BPE fragmentation.
- **Pass Criterion**: >80% exact argument copy on held-out arithmetic prompts without regex heuristics.

### Milestone 4: Runtime Performance & Hardware Quantization (Batches 1141–1180)
- **Objective**: Implement INT8 and FP16 quantization for CPU and GPU, enabling sub-50ms conversational latency on modest laptop hardware.
- **Pass Criterion**: 2x inference throughput increase on CPU with <1% loss in accuracy.

### Milestone 5: Autonomous Multi-Turn Task Orchestration (Batches 1181–1220)
- **Objective**: Combine Document Q&A, Memory, and Tool Calling into an autonomous multi-turn assistant capable of executing complex user requests.
- **Pass Criterion**: Complete 5 distinct multi-turn user workflows without human intervention.
