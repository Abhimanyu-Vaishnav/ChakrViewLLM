# EXTERNAL VS INTERNAL COGNITION: ARCHITECTURAL BOUNDARIES (CR-001)

A foundational architectural requirement of ChakrView is the strict, measurable separation between **internal neural capabilities** and **external runtime augmentations**.

---

## 1. The Separation Contract

| Functional Capability | Internal Neural Cognition (The Brain) | External System Augmentation (The Tools) | Honest Evaluation Rule |
|---|---|---|---|
| **Mathematical Computation** | Autoregressive weight-driven arithmetic ($A + B = C$) inside transformer layers. | Calling Python `math.sqrt()` or a local bash calculator. | External calculator calls MUST NOT be reported as internal mathematical capability. |
| **Knowledge Retrieval** | Factual associations embedded directly in MLP weights $W_{gate}, W_{up}, W_{down}$. | RAG, vector database lookup, SQLite search, file reading. | Retrieved document text MUST NOT be reported as internally acquired memory. |
| **Working Memory** | Multi-head attention key-value activations across the 512-token context window. | External scratchpad files, SQLite session store, disk persistence. | External file persistence MUST NOT be reported as neural working memory. |
| **Logical Validation** | Contradiction detection via attention inhibition heads. | Formal SAT solver or Prolog runtime execution. | External solver verification MUST NOT be attributed to the neural core. |
| **Language Translation** | Direct cross-lingual alignment in embedding and transformer space. | External API or dictionary replacement. | Dictionary replacement MUST NOT be reported as multilingual neural intelligence. |

---

## 2. The 4 Mandatory Evaluation Operational Modes

To prevent misleading claims, all future ChakrView capability reports must evaluate and publish results across four explicitly isolated modes:

```
Mode 1: Neural-Only (Bare Transformer Inference)
   [User Query] -> [ChakrMicro Neural Core] -> [Final Response]
   * Zero external tools, zero RAG, zero persistent files.
   * Measures true learned intelligence.

Mode 2: Neural + Working Memory (Session State Persistence)
   [User Query] + [Local Session Context] -> [ChakrMicro Neural Core] -> [Response]
   * Measures multi-turn retention over bounded context windows.

Mode 3: Neural + Grounded Retrieval (Local RAG)
   [User Query] -> [Local Document Retriever] -> [Context Injection] -> [ChakrMicro]
   * Measures reading comprehension and context extraction.

Mode 4: Neural + Tool Execution (Augmented Agent)
   [User Query] -> [ChakrMicro generates Tool Call] -> [Local Tool Executes] -> [ChakrMicro parses Output]
   * Measures agentic tool-use and instruction adherence.
```

Publishing scores without explicitly specifying which of these four modes was active is strictly forbidden.

