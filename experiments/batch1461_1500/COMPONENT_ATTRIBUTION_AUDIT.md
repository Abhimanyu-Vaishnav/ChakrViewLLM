# Track F: Component Attribution Audit

## Batches 1461–1500: Disentangling Neural Weights from Deterministic Engineering

### 1. Track Objective

To maintain scientific rigor and prevent misleading claims, this audit traces the exact execution path across all major assistant capabilities. We map every user-facing capability to its true underlying technical component:
- **Neural Learned Parameters (ChakrMicro Weights)**
- **Deterministic Information Retrieval (BM25 Index)**
- **Extractive Answer Selection & Verification Logic**
- **Refutation & Abstention Heuristic Rules**
- **External Python Tool Execution & Sandbox**

---

### 2. Comprehensive Capability Attribution Matrix

| Functional Assistant Feature | Evaluated Benchmark Track | Primary Driving Component | Neural Contribution (%) | Deterministic / Heuristic Contribution (%) | Scientific Classification |
|---|---|---|---|---|---|
| **Next-Token Continuation** | Track A: Language Modeling | ChakrMicro Neural Weights | **100%** | 0% | **Demonstrated (Statistical Modeling)** |
| **Factual QA / Architecture Recall** | Track B: Factual Grounding | BM25 Index + Extractive Span Selector | **0%** | **100%** | **Deterministic Tool-Driven** |
| **Citation Attribution & Provenance**| Track B & Controls | Filepath & Section Header Metadata | **0%** | **100%** | **Deterministic Metadata Rule** |
| **Arithmetic / Calculation** | Track C: Mathematics | None (Model Fails; Needs Python) | **0%** | 0% (Fails without Python) | **Not Demonstrated Neurally** |
| **Code Generation / Synthesis** | Track D: Coding | None (Model Fails; Needs Templates) | **0%** | 0% (Fails without templates) | **Not Demonstrated Neurally** |
| **Paraphrase Robustness** | Track E: Generalization | BM25 Inverted Index Token Matching | **0%** | **100%** | **Deterministic Lexical Overlap** |
| **Unanswerable Abstention** | Track Controls: Abstention | Score Threshold Gate ($\text{BM25} < \tau$) | **0%** | **100%** | **Deterministic Threshold Heuristic** |
| **False-Premise Refutation** | Track Controls: Refutation | Refutation Policy Regex & Fact Alignment | **0%** | **100%** | **Deterministic Rule Pipeline** |

---

### 3. Detailed Component Deep-Dive

#### 1. Why Standalone ChakrMicro Cannot Perform Factual QA
ChakrMicro is an indigenous compact transformer pre-trained autoregressively on technical documents. While it builds strong local transition probabilities (Track A), its internal weight representation does not store factual key-value pairs (Track B). When stripped of retrieval context, it emits repetitive phrases. All factual utility in the ChakrView assistant originates from reading retrieved text in the prompt context.

#### 2. The Nature of the Refutation and Abstention Capabilities
In Track Controls, the system achieved:
- **100.0% Clean Abstention (20/20)** on out-of-domain unanswerable queries (e.g., speed of light, Gold atomic mass, Sydney Opera House).
- **50.0% Explicit Refutation (10/20)** on false premise queries.

Tracing the code in `chakrview/knowledge/refutation.py` and `chakrview/knowledge/assistant.py` confirms:
1. Abstention occurs because the BM25 retrieval score does not pass the minimum relevance threshold $\tau$, causing the Python runtime to return the standard abstention message. The neural weights are never involved in this decision.
2. Refutation occurs because deterministic regex rules detect contradictory numbers or values against retrieved chunk attributes. Where regex rules match, refutation triggers; where they don't, the assistant defaults to answering or abstaining.

#### 3. Mathematical and Coding Reality
There is zero neural calculation or code synthesis occurring in the ChakrMicro weights. Both baseline and candidate scored 0.0% on arithmetic and functional code tests. Any future arithmetic or coding feature must be explicitly classified as external Python execution.

---

### 4. Integrity Assertion
> **Auditor Rule:** No technical documentation, README, or engineering report in ChakrView may describe factual answering, abstention, refutation, or calculation as "neural intelligence" or "learned reasoning." These are deterministic engineering capabilities wrapping a statistical token predictor.
