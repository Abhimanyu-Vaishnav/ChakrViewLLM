# ChakrView Step 51: Resource Constraints & Edge Profile Analysis

- **Version**: 1.0.0
- **Scope**: Step 51 — Coding Language Acquisition & Project Arena Foundation
- **Execution Target**: Low-end workstations, edge devices, and Raspberry-Pi-class SBCs
- **Primary Constraint**: Zero GPU requirement; CPU-first architecture

---

## 1. Hardware Feasibility & Low-Resource Profile

ChakrView's non-negotiable architectural requirement is seamless operation on low-power, legacy, or edge CPU architectures (including ARM Cortex-A72/A76, Raspberry Pi 4/5, and older x86 Intel Core / AMD CPUs).

| Resource Dimension | ChakrMicro Baseline | Step 51 Coding Model | Raspberry Pi 4 (4GB) Feasibility |
|:---|:---:|:---:|:---:|
| **Model Parameters** | 3,443,136 | 3,443,136 | Fully fits in L3/RAM |
| **Model File Size (FP32)** | $13.77\text{ MB}$ | $13.77\text{ MB}$ | Trivially stored on microSD/eMMC |
| **Static RAM Footprint** | $\approx 220\text{ MB}$ (PyTorch base) | $\approx 230\text{ MB}$ | $< 6\%$ of 4GB RAM |
| **Arena Peak RAM (RSS)** | $\approx 250\text{ MB}$ | $\approx 280\text{ MB}$ | Safe with zero swap pressure |
| **Max Context Window** | 512 tokens | 512 tokens | Negligible KV cache footprint ($< 1\text{ MB}$) |
| **Generation TTFT** | $7.5\text{ ms}$ | $\approx 7.8\text{ ms}$ | Real-time response |
| **Throughput (CPU)** | $43.2\text{ tok/sec}$ | $42.5\text{ tok/sec}$ | Comfortable interactive typing speed |
| **GPU / Accelerator** | None (0 MB VRAM) | None (0 MB VRAM) | 100% native CPU compatibility |

---

## 2. Dependency Audit & Cross-Device Portability

To guarantee that the Project Arena and coding evaluation subsystems run cleanly on ARM SBCs and low-end hardware, all introduced components use standard Python library modules:

- **`subprocess` / `os` / `shutil` / `tempfile`**: Native Python standard library.
- **`ast`**: Built-in Python AST compiler for syntax validation.
- **`pathlib` / `json` / `hashlib`**: Built-in Python standard library.
- **`pytest` / `unittest`**: Standard lightweight testing harness.
- **`torch`**: CPU-only PyTorch build.

**Prohibited Dependencies**:
- ❌ No CUDA, cuDNN, or ROCm runtimes.
- ❌ No Hugging Face `transformers`, `accelerate`, or `datasets`.
- ❌ No heavy LLM inference wrappers (vLLM, Ollama, llama.cpp, ONNX Runtime).
- ❌ No container or Docker daemon requirements (workspace isolation is purely OS process-based).

---

## 3. Sandboxed Execution Resource Caps

When the Project Arena executes tests on generated code, the following hard resource ceilings are strictly enforced:

1. **Wall-Clock Timeout**: $\le 5.0$ seconds per test run.
2. **Process Count**: Exactly 1 isolated worker subprocess.
3. **Environment Isolation**: Parent environment variables stripped; only necessary standard system paths (`PATH`, `SYSTEMROOT`, `PYTHONPATH`) provided.
4. **Clean Workspace Teardown**: Temporary workspaces deleted automatically after evaluation or preserved in dedicated scratch directories for inspection.
