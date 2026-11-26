# NR-019: Generalization, Regressions and Latency Report

## 1. Generalization Across Parameter Regimes
- **In-Distribution (300 tasks)**:
  - System A (Pure): 2.33%
  - System D (Grounded): **38.33%**
  - System E (Cascade): **39.67%**
- **Out-of-Distribution (500 tasks, large numbers $100-500$, novel syntax)**:
  - System A (Pure): 2.80%
  - System D (Grounded): **57.80%**
  - System E (Cascade): **58.60%**

**Finding**: Unlike autoregressive generation which suffers severe degradation when moving out-of-distribution, source-grounded routing achieves **higher accuracy out-of-distribution** (58.6% vs 39.7%) because larger numerals and explicit mathematical structures provide stronger, more unambiguous lexical anchors for the grounding engine.

---

## 2. Conversational Regression Probes
Evaluating baseline Stage-D foundation model fluency on general conversational prompts:
- Prompt: `"Hello, who are you?"` $\to$ Returns fluent conversational text (`coherent: true`).
- Prompt: `"Aapka naam kya hai?"` $\to$ Returns fluent multilingual text (`coherent: true`).
- **UTF-8 Integrity**: 100% valid UTF-8, zero corrupted surrogate pairs or non-printable artifacts.
- **Model Checkpoint Integrity**: Canonical Stage-D weights were unmodified.

---

## 3. End-to-End Latency Profile

| Component | Latency per Turn | Share of Turn Time |
| :--- | :---: | :---: |
| **Deterministic Executor** | **0.001 ms** | $0.01\%$ |
| **Source Grounding & Normalizer** | **0.011 ms** | $0.14\%$ |
| **Neural Router Forward Pass (32 tokens)** | **8.06 ms** | $99.75\%$ |
| **Total End-to-End Hybrid Turn** | **8.08 ms** | $100.0\%$ |

**Finding**: The entire neuro-symbolic grounding and execution pipeline executes in **8.08 milliseconds**, easily sustaining $>120$ full requests per second on a single GPU.
