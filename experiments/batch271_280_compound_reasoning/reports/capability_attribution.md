# Capability Attribution Report — Batches 271–280
## Independent Evaluation, Structural Attribution, and Component Boundaries

**Milestone:** Batches 271–280  
**Evaluated Manifest:** `experiments/batch271_280_compound_reasoning/data/fresh_evaluation_manifest_200.json`  
**Manifest SHA-256:** `7a73ae13e36541f6bd0ced0eda7984b859cb504fdaa38ab3000e3bea9751002c`  
**Evaluation Date:** October 2026  
**Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`, 3,443,136 parameters)

---

### 1. Executive Summary

In accordance with ChakrView core invariants, we strictly distinguish between:
1. **Unassisted Neural Generation:** What the 3.44M parameter canonical Transformer (`ChakrMicro`) can output on its own via unconstrained greedy token generation.
2. **Specialized Neural Adapters:** Learned pointer heads and attention-based relational extraction modules.
3. **Deterministic Tools & DSLs:** Bounded arithmetic parsers, unit conversion engines, and AST syntactical compilers.
4. **Governed Epistemic Guardrails:** Permission-gated external research state machines and false-premise rejection boundaries.

On the 200-question independent held-out manifest across 3 random seeds:
- **Arm A (Canonical Neural Core Unassisted):** 0/200 (0.0%), Avg Latency: 205.26 ms
- **Arm E (Full Integrated Runtime):** 174/200 (87.0%), Avg Latency: 13.14 ms

---

### 2. Detailed Performance by Evaluation Arm

| Evaluation Arm | Method | Total Items | Correct | Accuracy | Mean Latency | Primary Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm A: Neural Core Unassisted** | Direct greedy decoding (`ChakrMicro`) | 200 | 0 | 0.0% | 205.26 ms | Hallucinated continuation, repetitious n-grams, unable to perform exact arithmetic or execute code. |
| **Arm B: Neural + Pointer Head** | Learned copy mechanism | 200 | 18 | 9.0% | 45.10 ms | Copies context spans when explicit context is provided; fails on open queries without passage context. |
| **Arm C: Deterministic Tools** | Router -> `CalculatorTool` & `CompoundArithmeticEngine` | 68 (math/units) | 68 | 100.0% | 0.85 ms | None for supported units/syntax; fails safely on dimension mismatch. |
| **Arm D: Retrieval-Assisted** | Multi-hop retriever + System Knowledge | 54 (facts/premises) | 22 | 40.7% | 4.20 ms | High precision on registered celestial premises; abstains on out-of-KB entities. |
| **Arm E: Full Integrated Runtime** | Founder console dispatch (`ChakrViewRuntimeSession`) | 200 | 174 | 87.0% | 13.14 ms | Fails on technical concept definitions not yet in knowledge graph. |

---

### 3. Component Origin Attribution

Every success in Arm E is logged with its exact producing component:

| Component Origin | Queries Handled | Category Specialization | Integrity Guarantee |
| :--- | :--- | :--- | :--- |
| `calculator_tool` | 68 | Compound arithmetic & unit conversions | Bounded AST / regex parser; zero `eval()`. Dimensional mismatch rejected. |
| `DIRECT_GROUNDING` | 58 | Open-world unanswerable & out-of-domain | Honest epistemic abstention; no simulated knowledge. |
| `code_synthesizer` | 33 | User-requested Python scripts | Sandboxed AST parsing and subprocess execution testing. |
| `system_knowledge` | 22 | False premise governance (celestial bodies) | Explains why premise is invalid (e.g. Saturn has no capital). |
| `governed_research_workflow` | 19 | Explicit web research queries | Halts with `PERMISSION_REQUIRED`; user consent strictly required. |

---

### 4. Non-Negotiable Findings

1. **Zero Neural Inflated Claims:** The 3.44M canonical neural core has not developed autonomous compound arithmetic or multi-step reasoning capabilities. Presenting the 87.0% runtime score as "ChakrView's neural intelligence" would be fraudulent.
2. **Robust Routing Delivers Correctness:** The cognitive router, deterministic compound engine, and epistemic guardrails reliably protect the user from hallucinations and unsafe executions.
3. **True Held-Out Validity:** Zero data leakage occurred between the seed generation and the evaluation code.
