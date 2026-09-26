# ChakrView Step 4.1: Environment Verification

**Document Version**: 1.0.0  
**Phase**: Step 4.1 — Environment Verification  
**Status**: VERIFIED & REPRODUCIBLE  

---

## 1. System & Hardware Specifications

| Component | Measured Specification | Source / Tool |
| :--- | :--- | :---: |
| **Operating System** | Microsoft Windows 11 Enterprise (Build 26200) | `platform.platform()` |
| **Processor (CPU)** | 13th Gen Intel(R) Core(TM) i9-13900H (14 Cores / 20 Threads, up to 5.4 GHz) | `Win32_Processor` |
| **System Memory (RAM)** | 32.0 GB DDR5 ($33,177,868\text{ KB}$ Total, $> 17\text{ GB}$ Available) | `Win32_OperatingSystem` |
| **Graphics Hardware (GPU)** | NVIDIA RTX A1000 6GB Laptop GPU & Intel Iris Xe Graphics | `Win32_VideoController` |
| **Execution Policy** | Pure CPU Execution (Intentional Zero GPU Dependency) | Architectural Invariant |

---

## 2. Software Runtime & Scientific Backends

| Package / Runtime | Measured Version | Location / Details |
| :--- | :---: | :--- |
| **Python** | `3.14.7` | `D:\Project\ChakrView\.venv\Scripts\python.exe` (MSC v.1944 64-bit AMD64) |
| **PyTorch** | `2.14.0+cpu` | `D:\Project\ChakrView\.venv\Lib\site-packages\torch\__init__.py` |
| **CUDA in PyTorch** | `False` | Verified via `torch.cuda.is_available() == False` |
| **NumPy** | `2.5.3` | `D:\Project\ChakrView\.venv\Lib\site-packages\numpy\__init__.py` |
| **PyTest** | `9.1.1` | `D:\Project\ChakrView\.venv\Lib\site-packages\pytest\__init__.py` |
| **psutil** | `7.2.2` | Process RSS Memory Profiler |
| **BLAS / SIMD** | Intel MKL 2026.1 / AVX2 | Vectorized CPU execution enabled |

---

## 3. Environment Invariant Verification

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.__file__); print('CUDA:', torch.cuda.is_available())"
```
**Output**:
```text
2.14.0+cpu
D:\Project\ChakrView\.venv\Lib\site-packages\torch\__init__.py
CUDA: False
```

- [x] PyTorch CPU wheel properly installed into virtual environment.
- [x] Zero extraneous `.whl` files left in the workspace.
- [x] CPU tensor linear algebra fully operational (`a @ b` succeeds).
- [x] Virtual environment isolated at `D:\Project\ChakrView\.venv`.

---
*End of Environment Verification Document*
