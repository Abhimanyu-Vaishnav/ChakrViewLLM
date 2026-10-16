# Container-Runtime Capability Discovery & Host Audit

**Evaluation Date:** 2026-10-16  
**Host Environment:** Windows (Microsoft Windows 11 / Server Architecture: AMD64)  
**Execution Context:** Unprivileged Local User Environment  

---

## 1. Runtime Discovery & Inspection Findings

A systematic preflight check was executed to discover available OCI-compatible container runtimes and system virtualization capabilities:

| Runtime / Feature | Discovery Command | Exit Code | Result & Detailed Finding |
|:---|:---|:---:|:---|
| **Docker Engine / CLI** | `docker --version` | 1 | **NOT INSTALLED / NOT IN PATH**. Command not found. |
| **Podman CLI** | `podman --version` | 1 | **NOT INSTALLED / NOT IN PATH**. Command not found. |
| **nerdctl / containerd** | `nerdctl --version` | 1 | **NOT INSTALLED / NOT IN PATH**. Command not found. |
| **WSL2 Subsystem** | `wsl --status` | 1 | **NOT INSTALLED**. Windows Subsystem for Linux is not installed on this machine. |
| **Hyper-V / Windows Containers**| System Inspection | N/A | Unprivileged environment; unmanaged Windows containers unavailable. |

---

## 2. Security Capabilities Assessment

Because no OCI-compatible container runtime is installed or accessible on the host:
- **User Execution Level:** Current user does not have access to a container engine daemon or rootless container tooling.
- **Rootless Container Execution:** **UNSUPPORTED** on current host.
- **Kernel Network Namespace Isolation:** **UNAVAILABLE** (Windows host without WSL2/Docker cannot enforce Linux cgroups or `ip netns`).
- **Read-Only Root Filesystems:** **UNAVAILABLE**.
- **Container Memory/CPU Quotas (cgroups v2):** **UNAVAILABLE**.
- **Image Pulling & Verification:** **UNAVAILABLE & UNAUTHORIZED** (Mission safety rules prohibit arbitrary package/software downloads or machine reconfigurations).

---

## 3. Governed Policy & System Invariants

### Non-Negotiable Rule: No Pretend Sandboxes
Under Mission Rule 3 and Non-Negotiable Safety Rules:
> *"If no suitable runtime is available, implement a well-defined disabled/unavailable state and provide exact setup requirements. Do not substitute an ordinary subprocess and call it an OCI sandbox."*

### Implementation Directives:
1. **Explicit `OCI_RUNTIME_UNAVAILABLE` State:**  
   The container execution backend (`OciContainerBackend`) must discover runtime availability dynamically. If neither `docker` nor `podman` is present, it must return an explicit `OCI_RUNTIME_UNAVAILABLE` status and reject arbitrary execution requests.
2. **Fail-Closed Arbitrary Execution Policy:**  
   When OCI isolation cannot be verified, arbitrary code execution remains strictly disabled (`allow_arbitrary_code=False`).
3. **Defense-in-Depth Host Fallback Tier:**  
   For bounded internal tasks where code execution is authorized via explicit operator configuration, the system falls back to the static-AST inspected, ephemeral-workspace `SandboxedPythonTool` implemented in Batches 1541–1580, while clearly marking it as a **host-restricted subprocess fallback**, NOT an OCI container sandbox.

---

## 4. Setup Requirements for Production OCI Isolation
To enable full OCI container isolation in production:
1. Install Docker Desktop or Podman Desktop with WSL2 backend on Windows, or deploy ChakrView on a Linux host (kernel 5.15+ with cgroups v2 enabled).
2. Configure a hardened, minimal runner image (e.g. `python:3.11-alpine` or `gcr.io/distroless/python3-debian12`).
3. Grant the runtime execution privileges without `sudo` (e.g. rootless Podman or standard Docker group).
4. Verify execution of `docker run --network none --read-only --user 1000:1000 --cap-drop ALL --memory 256m ...`.
