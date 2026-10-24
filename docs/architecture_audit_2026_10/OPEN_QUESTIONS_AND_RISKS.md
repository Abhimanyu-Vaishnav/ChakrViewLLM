# Open Questions and Technical Risks
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. Primary Technical Risks

### Risk 1: Model Capacity Limits of 3.4M Parameters
- **Description**: ChakrMicro v0.1 has $3,443,136$ parameters. In modern LLM literature, models under 50M parameters rarely develop robust generalized language reasoning or complex multi-step world modeling.
- **Impact**: The model may excel at narrow syntactic pattern matching and span pointing, but fail completely at open-domain freeform dialogue without falling back to symbolic templates.
- **Mitigation**:
  - Keep neural and cognitive boundaries strictly modular. Use the neural core for fast, low-latency representation extraction and token pointing, while delegating planning and logic to symbolic engines.
  - Plan a scaled variant (e.g. ChakrMicro-15M or ChakrMini-50M) once the small architecture proves its architectural invariants.

### Risk 2: Query-Slot Entanglement in Multi-Field Pointer Extraction
- **Description**: Experiment 12 revealed that when extracting multiple fields (e.g. `entity`, `status`, `code`), the pointer repeatedly copies the first salient entity into all fields.
- **Impact**: Without explicit slot conditioning, the model cannot distinguish which field is currently being queried.
- **Mitigation**: Implement Milestone 1 (Field-Conditioned Pointer Attention) by projecting dedicated slot embeddings into the query attention head.

### Risk 3: Premature Distributed Complexity
- **Description**: The repository contains hundreds of files implementing federated consensus, cross-zone peering, and cognitive orchestration before single-device neural language capabilities are fully stabilized.
- **Impact**: High maintenance burden, code bloat, and false impressions of operational readiness.
- **Mitigation**: Freeze the distributed subsystem as an experimental research tier; prioritize single-device robustness on constrained hardware.

---

## 2. Open Architectural Questions

1. **Question 1: Pure Neural Generation vs. Pointer Extraction Tradeoff**
   - *Issue*: Should ChakrView continue pursuing open-vocabulary free-form JSON generation, or lean fully into span-pointer extraction for structured tasks?
   - *Recommendation*: Lean heavily into span-pointer extraction for domain entity grounding. A 3.4M model cannot memorize millions of domain entities, but it can learn to point to them in context.

2. **Question 2: In-Memory PyTorch vs. Native C++ / GGUF Runtime for Edge Devices**
   - *Issue*: PyTorch CPU execution carries runtime overhead (~100–200 MB base memory). For true edge deployments on 1–2 GB RAM devices, is a C++ or GGUF runtime necessary?
   - *Recommendation*: Preserve PyTorch for research and training, but export validated checkpoints to GGUF or a minimal C++ forward engine for standalone Tier 1 deployment.

3. **Question 3: Automated Self-Improvement Safety Boundary**
   - *Issue*: How can the proposed self-improvement loops operate without risking catastrophic forgetting or malicious training feedback loops?
   - *Recommendation*: Keep the $\Delta W \equiv 0$ neural baseline invariant strictly active. Self-improvement must update external memory, failure catalogues, and retrieval indices—never unattended in-place model weights.
