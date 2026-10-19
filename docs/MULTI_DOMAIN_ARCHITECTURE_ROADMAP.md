# ChakrView Multi-Domain Architecture Audit & Roadmap

**Classification**: Architecture Specification & Subsystem Inventory  
**Target Horizon**: v0.6 Hardening $\to$ v0.7 Coding Specialization $\to$ v1.0 November Release

---

## 1. Executive Summary

ChakrView is an indigenous modular AI platform designed around a **compact from-scratch neural core (`ChakrMicro`)** surrounded by governed cognitive, retrieval, memory, and verification layers. 

Coding is the **first specialization**, not the sole future domain. The architectural goal is to support domain-specific fine-tuning, adapter specialization, persistent memory, retrieval, and hardware adaptation **without destabilizing the frozen neural core**.

---

## 2. Subsystem Matrix: Implemented vs. Planned

| Architectural Subsystem | Current Repository Implementation Status | Existing File / Module | Future Requirements for v1.0 |
| :--- | :---: | :--- | :--- |
| **Base Model & Tokenizer Compatibility** | **IMPLEMENTED & TESTED** | `chakrview/brain/model.py`, `chakrview/tokenizer/` | Strict SHA-256 pre-verification, parameter shape checks, vocabulary size enforcement. |
| **Unified Runtime Adapter** | **IMPLEMENTED & TESTED** | `chakrview/runtime/neural_runtime_adapter.py` | CPU greedy & sampling generation, $\Delta W = 0$ post-execution validation. |
| **Hardware Discovery & CPU Fallback** | **IMPLEMENTED & TESTED** | `chakrview/runtime/hardware.py` | Host profiling (cores, RAM, OS), graceful CPU fallback when accelerators are absent. |
| **Retrieval & Document Ingestion** | **IMPLEMENTED & TESTED** | `chakrview/runtime/retrieval.py`, `assistant_cli.py` | BM25 lexical + semantic hybrid indexing, source citations. |
| **Session Memory & Evidence Precedence**| **IMPLEMENTED & TESTED** | `chakrview/runtime/conversation_memory.py` | Task-scoped working memory, verified external evidence precedence. |
| **Sandboxed Coding Test Execution** | **IMPLEMENTED & TESTED** | `evaluation/coding_suite/run_coding_evaluation.py` | Time-limited execution, sandboxed builtins (no filesystem/network access). |
| **Domain-Specific Adapter / LoRA Layer** | **PLANNED** | Concept documented; canonical core currently frozen | Low-rank parameter adapter matrices to isolate coding tuning without weight pollution. |
| **Controlled User-Feedback Ledger** | **EXPERIMENTAL / PROTOTYPED** | `CandidateLearningLedger` in `governed_tool_router.py` | Explicit user consent gate required before recording feedback tuples. |
| **Durable Task Recovery & Journaling** | **EXPERIMENTAL** | `chakrview/runtime/recovery.py` | Disk-backed checkpoint journaling for long-running workflows. |
| **GGUF Export & Quantization Pipeline** | **PLANNED (BLOCKED)** | `docs/GGUF_FEASIBILITY.md` | Standard llama.cpp tools lack custom ChakrMicro topology support (RoPE + SwiGLU 192-dim). |
| **Distributed / Multi-Node Execution** | **PLANNED / STUBBED ONLY** | `chakrview/runtime/oci_container_backend.py` | Mock wire interfaces exist; physical multi-node networking is **not verified**. |

---

## 3. Epistemic Separation: Training vs. Retrieval/Memory

To protect the integrity of the neural core and prevent silent model drift or hallucination:
1. **Weight Updates ($\Delta W \neq 0$)**:
   - Model weights can ONLY be updated via explicitly authorized offline training runs with versioned dataset manifests, pre-registered random seeds, and cryptographic checkpoints.
   - User interactions or internet-retrieved text **MUST NEVER** automatically mutate canonical weights during inference.
2. **Retrieval & Epistemic Memory**:
   - Ingested text, web sources, or user corrections are stored strictly in bounded memory stores or retrieval indexes (`WorkingMemory2`, BM25 lexical indices).
   - Inquiries requiring real-time knowledge must cite external evidence directly.

---

## 4. Hardware Adaptation: Real Capabilities vs. Limits

- **x86_64 CPU (Supported & Verified)**:
  - Multi-threaded forward passes via PyTorch CPU backend.
  - Tested throughput: ~136 tokens/second for 32-token generation.
  - Memory: ~271 MB process RSS.
- **CUDA / Apple MPS (Experimental Fallback)**:
  - System accurately queries `torch.cuda.is_available()` and detects GPU models.
  - If CUDA libraries are incomplete or the environment lacks GPU binaries, the runtime fails closed and defaults to CPU execution.
  - Full GPU tensor offloading and automated multi-GPU parallelism are **not verified** for production release.
