# ChakrView Empirical Capability Matrix (Hardened v0.7)

**Classification**: Grounded Empirical Evaluation Matrix  
**Evaluated Artifacts**: Canonical Pretrained Step 10,000 Checkpoint (`96357705...e9f471ee`) & Candidate Step-100 SFT V3 Pilot Checkpoint (`bfde007d...d1879`)  
**Scope**: All results reflect inference-only evaluations under deterministic greedy decoding.

---

## 1. Capability Scorecard & Evidence

| Capability Track | Evaluated Checkpoint | Benchmark / Split Source | Sample (N) | Measured Performance | Operational Evaluation Type | Truthful Epistemic Assessment |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Native Text Continuation** | Step 10,000 (Base) | Frozen Eval v2.0 (`CONT-01..05`) | 5 | **100.0% fluent domain text** | Autoregressive forward pass | **STRONG**: Fluent statistical continuation of learned domain prose. Exhibits loop collapse on unconstrained prompts exceeding 48 tokens. |
| **Context-Grounded Copying** | Step 10,000 & Step 100 SFT | Copy Capability Gate | 20 | **0/20 (0.0%)** exact match; **0/20 (0.0%)** normalized | Greedily decoded string match | **UNLEARNED**: The model does not copy arbitrary target entities from context. Cued by prompt framing rather than attention routing. |
| **Coding Completion & Syntax** | Step 10,000 (Base) | Coding Benchmark v0.7 (`coding_tasks.json`) | 6 | **2/6 (33.3%)** syntax valid; **1/6 (16.7%)** test pass | Sandboxed compilation & test execution | **ELEMENTARY**: Emits basic function definitions and assert statements; fails multi-line algorithmic logic. |
| **Instruction Following** | Step 10,000 & Step 100 SFT | SFT V3 Validation (`validation.jsonl`) | 40 | **0/40 (0.0%)** exact match | Deterministic exact match | **UNLEARNED**: Pretrained causal weights without instruction alignment; prompt boundaries cue pretraining continuation attractors. |
| **Structured Output (JSON)** | Step 100 SFT Candidate | SFT V3 Comparative Evaluator | 30 | **0/30 (0.0%)** exact match; 15%–90% code attractor | Schema parser + exact match | **WEAK**: Model learns superficial formatting (braces, quotes) but cannot populate ground-truth entity values. |
| **Retrieval-Grounded QA** | Hybrid Retrieval Subsystem | Local Knowledge Assistant (`assistant_cli`) | Multi-doc corpus | **High precision with citations** (0.875 conf) | BM25 lexical + semantic indexing | **FUNCTIONAL (RUNTIME)**: Factual answers with verbatim citations are provided entirely by the governed retrieval engine, not neural weights. |
| **Mathematical Reasoning** | Step 10,000 (Base) | Frozen Eval v2.0 (`ARITH-01..05`) | 5 | **0/5 (0.0%)** accuracy | Deterministic substring check | **ABSENT IN NEURAL CORE**: Arithmetic requires tool routing (`safe_calculator.py`). Autoregressive weights do not calculate arithmetic operations. |
| **Humor & Pragmatic Understanding** | Step 10,000 (Base) | Informal qualitative probes | Qualitative | **0.0%** intentional humor | Manual inspection | **ABSENT**: Emits statistical prose continuation without pragmatic awareness or comedic intent. |

---

## 2. Epistemic Separation & Overclaim Corrections

1. **Copying Failures**:
   - The 0/20 copying result does **not** prove that the 4.5M transformer architecture lacks theoretical capacity for induction heads. It proves only that 10,000 pretraining steps and 100 SFT pilot steps failed to develop those circuits under multi-task schema noise.
2. **Attractor Dynamics**:
   - Dialog framing (`User: ... Assistant:`) reduces the doctest code attractor from 90.0% to 15.0%, but does not eliminate all pretraining attractors (e.g., `raise ValueError(...)`).
3. **Retrieval vs. Neural Core**:
   - High-confidence citation-backed answers in the assistant CLI originate from the BM25 retrieval index and must never be attributed to learned neural factual memory.
