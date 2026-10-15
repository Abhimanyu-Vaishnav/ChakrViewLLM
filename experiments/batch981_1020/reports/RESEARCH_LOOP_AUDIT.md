# WP 987–992: Research Loop Audit & Loop Detection

## 1. Loop Detection: The Plan DSL Optimization Trap

An audit of Batches 741 through 980 reveals that research has been trapped in a **recursive local optimization loop around a synthetic DSL syntax**:

```
[Batches 741–780]
Autoregressive Plan DSL: Raw model fails syntax (0% valid, loops forever).
       │
       ▼
[Batches 781–820]
Add Grammar-Constrained Decoding: Syntax validity reaches 52.9%, but exact match is 8.6%, copy accuracy is 0/42.
       │
       ▼
[Batches 821–860]
Add Token-Level Pointer-Generator: Fails due to BPE subword splitting (exact match still 8.6%).
       │
       ▼
[Batches 861–900]
Add Span-Level Grounding: Span accuracy reaches 32.9%, but raw DSL generation produces 0/70 exact match.
       │
       ▼
[Batches 901–940]
Add Unified Plan Template Materializer: Syntax validity reaches 100%, but exact match drops to 2.9% due to topology mismatch.
       │
       ▼
[Batches 941–980]
Narrow to 2-step Arithmetic: Exact match reaches 36.4% on a narrow synthetic subset using positional heuristics.
```

---

## 2. Identified Research Vulnerabilities

1. **Deterministic Scaffolding Compensating for Model Limits**:
   The jump from 0/70 to 70/70 syntax validity in Batch 901–940 was achieved not because the neural core learned the grammar, but because `UnifiedPlan.materialize_dsl()` deterministically formatted lines and forced syntax validity.
2. **Synthetic Dataset Narrowing**:
   To show positive metrics in Batch 941–980, the evaluation scope was narrowed from a 70-item multi-topology benchmark to a 33-item 2-step arithmetic subset.
3. **Severe Core Disconnection**:
   The core transformer weights (`ChakrMicro`) remained frozen at step 10,000 for general reasoning, while peripheral classification and span heads were trained on dozens of synthetic examples.
4. **Zero Practical End-to-End Task Utility**:
   Despite 200 batches of work on planning, ChakrView currently cannot answer a user's question from a local text document, complete a multi-turn user conversation with grounded facts, or solve real mathematical problems without deterministic heuristic wrappers.
