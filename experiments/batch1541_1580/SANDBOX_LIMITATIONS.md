# Sandbox Limitations and Windows Isolation Constraints

## Batches 1541–1580: Documented Constraints of Native User-Space Execution

### 1. Fundamental Windows OS Constraints

1. **Absence of Linux-Style Namespaces & cgroups:**
   - On Windows, unprivileged Python processes cannot create mount namespaces, network namespaces, or PID namespaces without Docker, WSL2, or Windows Containers.
   - Consequently, low-level OS isolation primitives (e.g. `chroot`, `unshare`, `pivot_root`, `seccomp-bpf`) are unavailable in pure Python running on Windows.
2. **Memory Limit Enforcement:**
   - In standard user-mode Windows without Windows Job Objects (`win32job`), memory ceilings cannot be strictly enforced at the kernel level without third-party C extensions or pywin32 bindings.
3. **Network Isolation:**
   - Pure subprocess spawning cannot completely block outbound socket connections at the kernel firewall level without administrator privileges or dedicated Windows Firewall rules.

---

### 2. Operational Rules & Fail-Closed Behavior

Because true virtualization-grade isolation requires hypervisor or container support:
1. **Arbitrary Code Execution Disabled by Default:**
   `SandboxedPythonTool` operates with `allow_arbitrary_code = False`. Any attempt to execute unvetted arbitrary scripts is rejected with `SECURITY_VIOLATION`.
2. **Restricted AST Allowlist:**
   When bounded scripts are evaluated, the tool parses code through an AST inspector and rejects any attempt to import `socket`, `http`, `urllib`, `requests`, `subprocess`, `ctypes`, `os`, or `shutil`.
3. **Required Infrastructure for Full Sandboxing:**
   To safely enable unconstrained multi-tenant Python execution in future milestones, ChakrView must be deployed alongside a dedicated Docker daemon, a Firecracker microVM pool, or a gVisor sandboxed runner.
