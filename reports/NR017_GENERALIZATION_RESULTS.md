# NR-017 Phase 6 — Held-Out Generalization Results Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Sealed Benchmark:** `data/stage_g_017/stage_g_sealed_benchmark_400.json` (400 tasks, 10 families, 3 difficulty levels)  
**Detailed Breakdown Artifact:** `data/stage_g_017/detailed_generalization_breakdown.json`

---

## 1. Executive Summary & Comparative Generalization Verdict
All three models (Condition A Direct, Condition B Explicit Trace, Condition C Compact Trace) were evaluated across 400 sealed tasks to analyze family-wise and difficulty-wise generalization.

### **Key Scientific Finding:**
**NO EVIDENCE THAT STATE-TRACE SUPERVISION UNLOCKS REASONING CIRCUITS**
Across 10 distinct task families, performance remained functionally indistinguishable across all three conditions:
- **Direct Answer (Condition A):** Overall = **$16.8\%$**, Level 1 = $17.9\%$, Level 2 = $16.5\%$, Level 3 = $15.8\%$.
- **Explicit Trace (Condition B):** Overall = **$17.0\%$**, Level 1 = $17.9\%$, Level 2 = $15.8\%$, Level 3 = $17.3\%$.
- **Compact Trace (Condition C):** Overall = **$17.5\%$**, Level 1 = $16.4\%$, Level 2 = $20.3\%$, Level 3 = $15.8\%$.

---

## 2. Family-Wise Accuracy Comparison (40 Tasks per Family)

| Task Family | Family Type | Direct Answer (Cond A) | Explicit Trace (Cond B) | Compact Trace (Cond C) | Capability Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`algorithmic_compositions`** | Logic / Parity | **$100.0\%$ (40/40)** | **$100.0\%$ (40/40)** | **$100.0\%$ (40/40)** | Trivial boolean parity check (`x*2 is even -> True`) mastered 100% |
| **`arithmetic_operations`** | Multiplication | **$0.0\%$ (0/40)** | **$0.0\%$ (0/40)** | **$0.0\%$ (0/40)** | Multi-digit arithmetic fails across all formats |
| **`conditional_branching`** | Branch logic | **$0.0\%$ (0/40)** | **$0.0\%$ (0/40)** | **$0.0\%$ (0/40)** | Threshold comparison and branching fails |
| **`list_transformations`** | Reversal / Max | $2.5\%$ (1/40) | $2.5\%$ (1/40) | $2.5\%$ (1/40) | Near-total failure on list mutations |
| **`nested_operations`** | Parentheses | $0.0\%$ (0/40) | $2.5\%$ (1/40) | $2.5\%$ (1/40) | Minimal difference |
| **`sequential_state_updates`** | State accumulator | $2.5\%$ (1/40) | $0.0\%$ (0/40) | $0.0\%$ (0/40) | Multi-step accumulator failure |
| **`stack_operations`** | Push / Pop | $7.5\%$ (3/40) | $7.5\%$ (3/40) | $10.0\%$ (4/40) | Shallow stack tracking |
| **`string_transformations`** | String casing / rev | **$50.0\%$ (20/40)** | **$50.0\%$ (20/40)** | **$50.0\%$ (20/40)** | Exact split: 100% in-distribution, 0% OOD |
| **`variable_binding`** | Slot binding | $0.0\%$ (0/40) | $5.0\%$ (2/40) | $0.0\%$ (0/40) | Statistically negligible difference ($\pm 2$ tasks) |
| **`variable_renaming`** | Dependencies | $5.0\%$ (2/40) | $2.5\%$ (1/40) | $10.0\%$ (4/40) | Minor fluctuations |

---

## 3. Difficulty-Wise Breakdown
- **Level 1 (Single operation, 134 tasks):** Direct = $17.9\%$, Explicit Trace = $17.9\%$, Compact Trace = $16.4\%$.
- **Level 2 (Multiple dependent ops, 133 tasks):** Direct = $16.5\%$, Explicit Trace = $15.8\%$, Compact Trace = $20.3\%$.
- **Level 3 (Compositional interacting states, 133 tasks):** Direct = $15.8\%$, Explicit Trace = $17.3\%$, Compact Trace = $15.8\%$.

**Deduction:** State-trace supervision did not alleviate the degradation on complex Level 3 tasks. The network's error modes stem from token hallucination within the state equations themselves.
