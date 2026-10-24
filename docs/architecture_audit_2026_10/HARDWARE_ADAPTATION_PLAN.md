# Hardware Adaptation Plan: Progressive Execution Across Device Classes
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. Executive Principles

ChakrView must never assume high-end server hardware or dedicated GPU accelerators as prerequisites for operation. It is built as a **CPU-first, resource-elastic intelligence engine** capable of executing across three distinct hardware tiers:
1. **Tier 1 (Constrained Legacy / Edge)**: Dual-core CPU, 2–4 GB System RAM, no GPU, spinning HDD or slow eMMC.
2. **Tier 2 (Standard Consumer / Laptop)**: 4–8 core modern CPU, 8–16 GB System RAM, integrated or low-end GPU, NVMe SSD.
3. **Tier 3 (High-End Workstation / Server)**: 12+ core CPU, 32+ GB RAM, dedicated CUDA/ROCm/Metal accelerator.

---

## 2. Hardware Profile Matrix & Dynamic Allocation Policies

| Hardware Parameter | Tier 1: Constrained Edge | Tier 2: Standard Consumer | Tier 3: Workstation / Server |
| :--- | :--- | :--- | :--- |
| **Target Host RAM** | $2\text{ GB} - 4\text{ GB}$ (Low) | $8\text{ GB} - 16\text{ GB}$ (Standard) | $32+\text{ GB}$ (High) |
| **Max Process Memory Budget** | $\le 512\text{ MB}$ | $\le 2\text{ GB}$ | $\le 8\text{ GB}$ |
| **Max Sequence Context Window** | 512 tokens | 1,024 tokens | 2,048 tokens (Full model capacity) |
| **KV-Cache Eviction Strategy** | Sliding Window (256 tokens) + Aggressive FP16/INT8 quant | Static Allocation FP32 / FP16 | Unbounded Full-History KV-Cache |
| **Inference Concurrency** | Single-threaded serial ($1$ worker) | $2 - 4$ concurrent worker threads | $8+$ concurrent worker threads |
| **Model Weights Precision** | INT8 / Dynamic CPU Quantization | FP32 (Native PyTorch CPU) | FP16 / BF16 or TensorRT / ONNX |
| **Optional Modules Enabled** | Neural Core + FSM Controller only | + PPB Memory + Local BM25 Retrieval | + Full Dialectical Reasoner + Distributed Nodes |
| **Task Checkpointing Policy** | Immediate atomic sync after every subtask | Batch checkpointing (every 5 subtasks) | Asynchronous background journaling |

---

## 3. Dynamic Degradation & Resource Throttle Engine

The runtime engine in `chakrview/runtime/adaptive_engine.py` continuously samples system health metrics:
1. **Host Memory Pressure Check**:
   - If available RAM falls below $350\text{ MB}$, trigger emergency eviction:
     - Unload background semantic retriever indices.
     - Compress or evict non-essential conversation history from KV-cache.
     - Reduce batch size to $1$.
     - Yield cooperative sleep slices to prevent OS swap thrashing.
2. **Thermal & Core Throttling**:
   - Limit OpenMP/MKL thread counts to $\min(\text{physical\_cores}, 4)$ on laptops to avoid thermal throttling.
3. **Graceful Degradation Reporting**:
   - If a prompt requests an operation exceeding the current device tier (e.g., loading an oversized multi-hop context into a 512-token budget), the engine must **never crash or throw uncaught OutOfMemory errors**.
   - Instead, it emits a structured limitation report:
     ```json
     {
       "status": "DEGRADED_EXECUTION",
       "reason": "CONTEXT_BUDGET_EXCEEDED",
       "action_taken": "TRUNCATED_TO_SLIDING_WINDOW_256",
       "confidence_penalty": 0.25
     }
     ```

---

## 4. Hardware Verification Test Suite

Every release gate must run reproducible hardware verification tests:
1. **Low-Memory Emulation Smoke Test**:
   - Run inference under a synthetic memory cgroup or process limiter ($512\text{ MB}$ ceiling). Verify process completes without SIGKILL or unhandled MemoryError.
2. **Pure CPU Invariant Test**:
   - Ensure `torch.cuda.is_available()` returns `False` or is completely bypassed, verifying that no CUDA kernel calls exist in the inference path.
3. **Latency Benchmarking across Batch Sizes**:
   - Benchmark generation time for 64, 128, and 256 generated tokens on CPU single-thread vs. multi-thread, establishing baseline tokens/sec.
