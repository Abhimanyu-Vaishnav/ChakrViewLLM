# Batch 159 — Integrated Capability & Failure Analysis
## Rigorous 15-Case Battery Evaluation, Latency Profile & Architectural Limitations

### 1. Benchmark Execution Summary
- **Execution Date**: 2026-10-13
- **Device**: CPU (Thread-bounded, hardware-adaptive fallback)
- **Model Checkpoint**: Canonical Step 10,000 (`checkpoints/training_checkpoint_step_10000.pt`)
- **Total Test Cases**: 15
- **Passed Cases**: 15 / 15 (100.0%)
- **Route Accuracy**: 15 / 15 (100.0%)
- **Component Attribution Accuracy**: 100.0%

### 2. Case-by-Case Benchmark Results

| Test ID | Case Description | Route Selected | Component Responsible | Answer Preview | Result |
|---|---|---|---|---|---|
| **TC-01** | Direct Conversational Greeting | `direct_chat` | `neural_core` | Neural generative response | **PASS** |
| **TC-02** | Grounded QA - Atomic Number | `knowledge_retrieval` | `deterministic_extractor` | `6` | **PASS** |
| **TC-03** | Grounded QA - Desert Location | `knowledge_retrieval` | `deterministic_extractor` | `Rajasthan` | **PASS** |
| **TC-04** | Absent Fact Abstention | `insufficient_evidence` | `fallback_policy` | "I do not have sufficient information..." | **PASS** |
| **TC-05** | Supplied Context Key-Value | `slot_extraction` | `deterministic_extractor` | `diamond` | **PASS** |
| **TC-06** | Supplied Context with Distractors | `slot_extraction` | `deterministic_extractor` | `Sanskrit` | **PASS** |
| **TC-07** | Supplied Context Absent Key | `slot_extraction` | `deterministic_extractor` | `Not mentioned in context.` | **PASS** |
| **TC-08** | Multilingual (Hindi) Grounded QA | `knowledge_retrieval` | `deterministic_extractor` | `तांबा` | **PASS** |
| **TC-09** | Grounded Entity Extraction (Crater) | `knowledge_retrieval` | `deterministic_extractor` | `Shackleton` | **PASS** |
| **TC-10** | Multi-Turn Dialogue Continuity | `direct_chat` | `neural_core` | Neural conversational response | **PASS** |
| **TC-11** | Cross-Session Isolation Verification | `direct_chat` | `neural_core` | Fresh session isolated | **PASS** |
| **TC-12** | Conflicting Evidence in Context | `slot_extraction` | `deterministic_extractor` | "Conflicting evidence detected..." | **PASS** |
| **TC-13** | Empty / Whitespace Query Handling | `direct_chat` | `neural_core` | Safe fallback response | **PASS** |
| **TC-14** | Long Input Context Truncation | `slot_extraction` | `deterministic_extractor` | `apple` | **PASS** |
| **TC-15** | Grounded Negative Query Abstention | `slot_extraction` | `deterministic_extractor` | `Not mentioned in context.` | **PASS** |

### 3. Latency & Resource Footprint
- **Deterministic Extraction Path**: $\approx 2.1 - 4.5\text{ ms}$ (Sub-5ms, zero tensor allocation overhead).
- **Retrieval + Extraction Path**: $\approx 3.2 - 6.8\text{ ms}$ (Hybrid BM25 + Hash vector search).
- **Neural Autoregressive Path**: $\approx 280 - 1500\text{ ms}$ (CPU forward passes for 32 tokens).
- **Memory Footprint**: Peak RAM < 350 MB; zero external database daemon required.

### 4. Known Architectural Limitations & Honesty
1. **Neural Extractive Copying**: The 3.44M parameter base core cannot yet reliably copy arbitrary context spans zero-shot. While deterministic extraction bridges this gap for structured QA, the neural core itself must eventually acquire in-context conditioning via continued pretraining (Stage D revived) or context-mixture instruction tuning.
2. **Dense Semantic Embeddings**: The reference vector retriever uses deterministic hash buckets. While effective for structural testing, it cannot perform true conceptual semantic matching (e.g., matching "feline" to "cat"). Upgrading this requires an indigenous dense embedding head.
