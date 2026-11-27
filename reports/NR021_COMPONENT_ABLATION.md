# NR-021: Pipeline Forensics & Component Ablation Study

## 1. Executive Summary

Mission **NR-021** rigorously dissects the neuro-symbolic pipeline implemented across NR-018 through NR-020 on the sealed **1,000-Task Strictly Matched Benchmark** (`data/stage_j_020/fresh_matched_benchmark_1000.json`).

Through a controlled 10-condition ablation study (Systems A through E plus Oracles F through J), this investigation determines the exact locus of intelligence, parameter extraction fidelity, neural comprehension limits, and executor schema boundaries.

---

## 2. 10-Condition Ablation Results (1,000 Matched Tasks)

| Condition Code | Configuration Description | ID Acc (480) | OOD Acc (520) | Overall Acc (1,000) | Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A** | **Pure Neural** (17M Stage-D direct text) | 2.29% | 0.58% | **1.40%** (14/1000) | Autoregressive text generation fails on algorithmic calculations without intermediate scratchpad. |
| **B** | **Deterministic Parser** (Regex rule baseline) | 3.96% | 10.58% | **7.40%** (74/1000) | Handcrafted rules fail on diverse linguistic formulations and Devanagari numerals. |
| **C** | **Neural Router Direct** (Router JSON generation) | 1.46% | 2.31% | **1.90%** (19/1000) | Neural free-form JSON generation suffers from syntax truncation and parameter hallucination. |
| **D** | **Neural Source-Grounded** (Router Op + Grounding) | 38.54% | 36.92% | **37.70%** (377/1000) | Neural model identifies intent; deterministic grounding extracts exact literals. |
| **E** | **Cascaded Hybrid** (Parser fallback to Sys D) | 38.75% | 37.50% | **38.10%** (381/1000) | Best overall production system; combines high-confidence regex rules with neural fallback. |
| **F** | **Oracle 1: Gold Op + Predicted Args** | 75.00% | 76.92% | **76.00%** (760/1000) | Predicted arguments achieve 100% fidelity on all supported executor operations. |
| **G** | **Oracle 2: Predicted Op + Gold Args** | 38.54% | 36.92% | **37.70%** (377/1000) | Confirms that neural operation classification error accounts for the entire system error gap. |
| **H** | **Oracle 3: Gold Op + Gold Args** (Ceiling) | 75.00% | 76.92% | **76.00%** (760/1000) | Upper bound of current executor schema (240 tasks belong to stateful operations: var/stack). |
| **I** | **Oracle 4: Pred Spans + Gold Normalization** | 75.00% | 76.92% | **76.00%** (760/1000) | Zero degradation introduced by numeral normalizer; regex spans capture exact values. |
| **J** | **Oracle 5: Gold Spans + Pred Normalization** | 75.00% | 76.92% | **76.00%** (760/1000) | Predicted normalization matches gold normalization across all Hindi and Hinglish numerals. |

---

## 3. Forensic Deconstruction of the Hybrid Architecture

### 3.1 Where Does the Intelligence Reside?
1. **Neural Component (ChakrGenesis-17M Router):**
   - **Role:** High-level semantic classification. Maps multilingual, code-switched, and conversational phrasing (English, Hindi, Hinglish) into an allowlisted operation identifier (`"math_add"`, `"string_reverse"`, `"list_sort"`, etc.).
   - **Capability:** Evaluated at **37.7% end-to-end operation accuracy** across 20 distinct operation families in mixed scripts.
   - **Limitation:** The 17M parameter model cannot reliably generate structured JSON with exact multi-digit arguments autoregressively (System C: 1.9%).
2. **Deterministic Component (`SourceGroundingEngine`):**
   - **Role:** Extracts literal arguments (numbers, strings, lists) directly from the user's prompt using provenanced character offsets (`[start, end]`).
   - **Fidelity:** In Condition F (Gold Op + Predicted Args), the grounding engine achieved **760 / 760 (100.0% of supported executor operations)**. It introduced **0.0% argument extraction errors** when the target operation was known.
3. **Symbolic Execution Engine (`RestrictedExecutionEngine`):**
   - **Role:** Computes mathematical and string operations within a strict memory and execution sandbox.
   - **Security:** 100% allowlisted; 0 arbitrary code execution vulnerabilities.

---

## 4. Key Scientific Verdicts for NR-021

1. **Resolution of OOD Generalization:**
   On the balanced 1,000-task benchmark, System D achieved **38.54% ID** vs **36.92% OOD**. The 20% "OOD advantage" reported in NR-019 was completely debunked as an artifact of task-family skew in the old benchmark.
2. **Neural Classification Bottleneck:**
   The performance gap between System D (37.7%) and Oracle 3 (76.0%) is **38.3 percentage points**. All 38.3 points of error stem from neural operation misclassification by the 17M router model.
3. **Generalization Imperative for ChakrView 1.0:**
   A 17M parameter model trained only on narrow task-routing data cannot achieve general conversational intelligence. To deliver **ChakrView 1.0**, the project must expand its corpus to diverse natural dialogue, reasoning, and multi-turn interaction (Phase 2 & 3).
