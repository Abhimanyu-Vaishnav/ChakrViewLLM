# Native Capability Benchmark Plan: Isolating Learned Neural Intelligence
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. The Core Scientific Invariant

Previous experiments (Experiments 07–12) demonstrated a fundamental risk: **symbolic scaffolding, regex templates, and hardcoded controllers frequently mask underlying neural deficiencies**.

To establish genuine scientific credibility, ChakrView must evaluate capabilities across five distinct, unblurred experimental tiers:
- **Tier A (Native Causal LM)**: Pure neural next-token prediction without external memory, tools, or decoders.
- **Tier B (Native LM + Neural Pointer)**: Neural model augmented strictly with the learned span-pointer head.
- **Tier C (Constrained Hybrid)**: Neural model + pointer guided by the deterministic finite-state schema controller.
- **Tier D (Diagnostic Oracle Upper Bound)**: System provided with ground-truth span locations to measure ceiling performance.
- **Tier E (Deterministic Rule Baseline)**: Transparent regex/heuristic baseline (serves as the non-neural floor).

---

## 2. Benchmark Suite Taxonomy

The benchmark consists of 10 targeted evaluation suites designed to probe specific failure modes:

| Suite ID | Evaluation Objective | Evaluation Dataset / Task Format | Metric & Scoring Formula | Disjoint Split Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **BENCH-01** | Next-Token Language Modeling | Real-world Stage C Wikipedia / Technical corpus | Perplexity ($\text{PPL} = \exp(\text{CrossEntropy})$) | Held-out document shards never seen in training |
| **BENCH-02** | Paraphrase & Meaning Robustness | Paired statements with varying surface syntax | Exact Match & Embedding Cosine Similarity | Novel vocabulary combinations |
| **BENCH-03** | Negation & Polarity Handling | Contexts containing "is not", "never", "disabled" | Polarity Inversion Accuracy | Contrastive synthetic pairs |
| **BENCH-04** | Instruction Following | Multi-step commands ("Return only X", "Format as Y") | Schema Compliance Rate & Keyword Containment | Held-out task instructions |
| **BENCH-05** | Context Retrieval & Distractor Resistance | Needle-in-a-haystack tasks with 10 distractor spans | Retrieval Precision @ 1 & Distractor Rejection Rate | Synthetic distractors with high token overlap |
| **BENCH-06** | Multi-Sentence Information Integration | 2-hop entity tracking across sentences | Multi-hop Exact Match | Disjoint entity pairs |
| **BENCH-07** | Multi-Step Symbolic Reasoning | Arithmetic chain evaluation ($A + B \times C$) | Step-by-step Intermediate Correctness | Unseen multi-digit operands |
| **BENCH-08** | Calibrated Uncertainty & Abstention | Questions whose answers are omitted from context | Abstention Rate (Emitting "none" or "unknown") | 50% missing-value test split |
| **BENCH-09** | Structured Schema Extraction | Extracting `{"entity": "...", "status": "...", "code": "..."}` | Exact Whole-Object Match (Obj EM) | Disjoint held-out entity identities (Test B) |
| **BENCH-10** | Code Syntax & Task Completion | Simple Python function synthesis with assertions | Pytest Pass Rate (0/1 binary execution) | New unit test specifications |

---

## 3. Strict Evaluation Integrity Rules

1. **Denominators are Inviolate**:
   - Every failure (syntax error, timeout, malformed string, infinite loop) remains in the denominator. Never exclude invalid outputs to inflate accuracy scores.
2. **Deterministic Reproducibility**:
   - Temperature is fixed to $0.0$ (greedy decoding) for capability benchmarks.
   - Global seed is fixed to $42$.
   - All evaluated examples, prompts, ground truths, and raw outputs are saved to a machine-readable CSV (`BENCHMARK_PER_EXAMPLE.csv`).
3. **Controller Disclosure**:
   - Whenever an FSM controller is active, report explicitly that structural validity is enforced by the controller and does not represent learned grammar induction by the neural weights.
