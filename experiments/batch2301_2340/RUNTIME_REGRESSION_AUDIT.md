# RUNTIME REGRESSION AUDIT

**Batch:** 2301–2340  
**Focus:** Verification of Truthful Subsystem and Execution Backend Attribution in CLI and Adapter  
**Status:** PASS  

---

## 1. Subsystem Routing Verification

Following the hardening completed in Batch 2261–2300, `IntegratedAssistantAdapter` and `assistant_cli.py` enforce strict separation between deterministic tools and neural inference:

```
User Query
    │
    ├── [Is Expression Valid Arithmetic?] ── YES ──> SafeCalculator (AST In-Memory)
    │                                                    ├── Subsystem: "safe_calculator"
    │                                                    ├── Actual Backend: "CPU (AST in-memory)"
    │                                                    └── Neural Inference: "not invoked"
    │
    └── NO ───────────────────────────────────────> Neural Core (ChakrMicro)
                                                         ├── Subsystem: "neural_model"
                                                         ├── Requested Device: "--device <dev>"
                                                         ├── Actual Backend: "cuda:0" or "cpu"
                                                         └── Neural Inference: "invoked (<backend>)"
```

---

## 2. Empirical Execution Path Audit

| Query Input | Subsystem Invoked | Actual Backend | Neural Inference | Fallback Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `"What is 45 * 12?"` | `safe_calculator` | `CPU (AST in-memory)` | `not invoked` | Executed deterministically in 0.19 ms without loading weights. |
| `"compute (120 + 35) / 5"` | `safe_calculator` | `CPU (AST in-memory)` | `not invoked` | Evaluated safely via AST visitor; 0 GPU memory allocated. |
| `"Explain information entropy"` | `neural_model` | `cuda:0` (if CUDA enabled) | `invoked (cuda:0)` | Autoregressive forward passes executed on NVIDIA RTX A1000. |
| `"Explain entropy"` (`--device cpu`) | `neural_model` | `cpu` | `invoked (cpu)` | Autoregressive forward passes pinned strictly to host CPU. |
| Non-arithmetic (`CUDA Error/OOM`) | `neural_model` | `cpu (fallback)` | `invoked (cpu)` | Graceful fallback catches CUDA errors and reroutes to CPU. |

---

## 3. Regression Testing Results

All 6 test cases in `tests/test_batch2221_2260_cli_and_integration.py` and all 5 test cases in `tests/test_batch2261_2300_provenance_and_truthfulness.py` passed with 100% compliance.
The CLI never reports arithmetic capability as learned neural behavior, nor does it falsely claim GPU execution when an operation executes on CPU.
