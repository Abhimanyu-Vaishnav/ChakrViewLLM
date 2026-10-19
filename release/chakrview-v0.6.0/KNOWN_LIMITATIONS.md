# ChakrView v0.6.0 Known Limitations & Capability Boundaries

**Classification**: Scientific Truthfulness & Epistemic Boundaries

---

## 1. Clear Separation of Neural Core vs Governed Runtime

Users and evaluators must clearly distinguish the different layers of ChakrView:

1. **ChakrMicro Neural Core (3.44M parameters, Step 10,000 Checkpoint)**:
   - A compact, from-scratch autoregressive language model trained purely on pretraining corpora.
   - Capable of fluent domain text continuation and statistical language modeling.
   - **NOT** an instruction-following assistant. It does not natively possess conversational alignment or multi-turn instruction adherence.
2. **Local Knowledge Assistant & Hybrid Retrieval Engine**:
   - Uses BM25 lexical search, semantic indexing, and grounded citation verification over ingested local documents.
   - When answering factual questions with citations (e.g. `ask "What is ChakrView?"`), the answers originate from the **governed retrieval pipeline**, not the neural weights.
3. **Structured Entity Extraction Status**:
   - Recent controlled experiments (SFT V3 pilot and Copy Capability Gate) established that the candidate model achieves **0/20 (0.0%)** exact context copying across tested prompts.
   - Switching to dialog formatting (`User: ... Assistant:`) suppresses code attractors (reducing doctest emission from 90% to 15%), but **does not grant native copying capability**.
   - Structured JSON schema decoders and prefix tries can constrain syntax, but cannot manufacture factual entity values that the underlying model has not learned to attend to.

---

## 2. Hardware Adaptation Boundaries

- **CPU-First Execution**: The neural inference pipeline is fully tested, optimized, and supported on x86_64 CPU hardware (Windows and Linux).
- **GPU / Accelerator Status**:
  - Hardware discovery accurately detects whether CUDA or MPS is available on the host.
  - End-to-end adaptive execution on NVIDIA/CUDA or Apple/MPS is currently an experimental fallback prototype. On systems without full CUDA libraries or with torch CPU-only builds, the system fails safely and defaults to native CPU inference.
  - Do not claim full automated distributed GPU cluster execution.

---

## 3. Generative Limitations & Failure Modes

1. **Repetition & Loop Collapse**:
   - Under unconstrained open-ended generation with high token budgets (>48 tokens), ChakrMicro can enter repetitive n-gram loops (e.g. `"the world of the world of..."`).
   - Repetition penalties (e.g. `repetition_penalty=1.2`) alleviate but do not eliminate this behavior.
2. **Context Budget Constraints**:
   - The canonical sequence length limit is strictly **512 tokens**.
   - Inputs exceeding 512 tokens are rejected or truncated to preserve numerical stability and memory bounds.
3. **Pretraining Code Attractor**:
   - When prompted with bare code transitions (such as `\nResponse: {"`), the model's pretraining weights have a strong propensity to generate Python unit test code (`{"__main__":\n import doctest`).
   - Using dialog framing (`User: ... Assistant:`) substantially suppresses this attractor (reducing doctest emission from 90.0% to 15.0% in Copy Capability Gate tests), though remaining code continuation attractors (such as `raise ValueError(...)`) may still occur. Dialog framing does not eliminate all attractors nor does it induce context copying.
