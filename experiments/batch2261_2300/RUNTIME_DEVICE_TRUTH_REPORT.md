# RUNTIME DEVICE TRUTH REPORT

**Batch:** 2261–2300  
**Topic:** Truthful Subsystem Execution Attribution & Hardware Backend Tracing  
**Modules Audited:** `chakrview/runtime/assistant_cli.py`, `experiments/batch2221_2260/integrated_assistant_adapter.py`  

---

## 1. The Attribution Problem: Why Reporting "CUDA" for Arithmetic Was False

In Batch 2221–2260, running `python -m chakrview.runtime.assistant_cli --device cuda ask "Calculate 16 * 16"` returned:
`Decision: CALCULATED | Device: CUDA | Latency: 0.18 ms | Answer: 256`

### Root Cause of the Misleading Label:
1. `Device: CUDA` merely reflected the user's input parameter `--device cuda` passed into `AdaptiveExecutionPolicy`.
2. The query was intercepted by `VerifiedToolRouter` and executed by `SafeCalculator`.
3. `SafeCalculator` is a pure in-memory Python AST evaluator that runs **entirely on the host CPU** using Python's standard `operator` functions.
4. **No neural model weights were loaded, no PyTorch tensors were allocated on CUDA, and no GPU kernels were launched.**
5. Reporting "Device: CUDA" falsely implied that a neural model on the GPU had computed the arithmetic.

---

## 2. Hardening and Truthful Subsystem Attribution

To enforce scientific truthfulness, `IntegratedAssistantAdapter` and `assistant_cli.py` were hardened in Batch 2261–2300 to report a structured **execution breakdown**:
- `subsystem`: Identifies the exact component that resolved the request (`safe_calculator` vs `local_knowledge_assistant` vs `neural_inference`).
- `actual_backend`: Identifies the physical hardware backend executing the code.
- `neural_inference`: Explicitly reports `not invoked` when neural generation was bypassed.

### Verified CLI Output Example:
```powershell
.venv_gpu\Scripts\python.exe -m chakrview.runtime.assistant_cli --device cuda ask "What is 500 - 234?"
```
```text
Decision: CALCULATED (confidence: 1.0) | Latency: 0.33 ms
Subsystem: safe_calculator | Actual Backend: CPU (AST in-memory)
Answer: 266 (calculated via safe_calculator)
```

---

## 3. Subsystem Execution & Attribution Matrix

| User Query Scenario | Routed Subsystem | Actual Hardware Backend | Neural Model Invoked? | Fallback Behavior |
| :--- | :--- | :--- | :---: | :--- |
| **Arithmetic (e.g. "500 - 234")** | `SafeCalculator` (AST) | **CPU (Host RAM)** | **No** (Bypassed) | Rejects disallowed syntax; no GPU fallback needed |
| **Local Knowledge QA (Document query)** | `LocalKnowledgeAssistant` (Hybrid BM25) | **CPU (Host RAM)** | **No** (BM25 + Lexical) | Abstains if no verified citation |
| **Direct Neural Evaluation (CPT evaluation)** | `ChakrMicro` (PyTorch) | **CPU or CUDA** | **Yes** | Graceful fallback to CPU if CUDA is unavailable |
| **Batched Forward Pass (Benchmarking)** | `ChakrMicro` (PyTorch) | **CUDA (RTX A1000)** | **Yes** | 5.52x faster on GPU; CPU fallback verified |

---

## 4. Subsystem Verification Checklist

1. **Truthfulness:** When arithmetic is solved by `SafeCalculator`, the CLI now explicitly reports `Subsystem: safe_calculator | Actual Backend: CPU (AST in-memory)`. It does NOT report CUDA.
2. **CPU Fallback:** When `--device cpu` is passed, all neural components are constrained to host CPU.
3. **CUDA Activation:** When batched model evaluation or training scripts run with `device="cuda"`, tensors are allocated in GPU VRAM (verified peak allocated VRAM: ~208 MB).
4. **Session Boundary:** `chakrview/runtime/session.py` was **not modified**.
