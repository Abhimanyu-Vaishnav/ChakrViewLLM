# Sandbox Security Model

## Batches 1541–1580: Threat Analysis, Isolation Facilities & Security Boundaries

### 1. Operating Environment & Threat Model

The host operating system for this ChakrView deployment is **Microsoft Windows**. In native Windows user-space without administrative privileges, lightweight containerization (e.g. Linux cgroups, namespaces, seccomp-bpf, chroot) is **not natively available** to standard Python worker processes without Docker, WSL2, or Windows Sandbox hypervisor integration.

#### Critical Security Rule:
> **A Python subprocess with a wall-clock timeout is NOT a secure sandbox.**  
> It cannot prevent arbitrary filesystem reads/writes within the user's permission boundary, host port probing, registry modifications, or process enumeration.

---

### 2. Sandbox Security Architecture & Controls

To safely evaluate bounded Python code execution, the `SandboxedPythonTool` enforces a **defense-in-depth, fail-closed isolation model**:

| Security Control | Enforced Mechanism | Implementation Details |
|---|---|---|
| **Filesystem Isolation** | Ephemeral Temporary Directory | Subprocess execution runs exclusively inside an isolated temporary directory created via `tempfile.TemporaryDirectory()`. The directory is securely shredded and unlinked upon completion. |
| **Execution Timeouts** | Wall-clock & Subprocess Timeout | Hard process termination enforced via `subprocess.Popen.communicate(timeout=2.0)`. Processes exceeding timeout are forcefully killed (`proc.kill()`). |
| **Output Size Clamping** | Stdout & Stderr Buffer Caps | Standard output and error buffers are capped at **64 KB** (`MAX_OUTPUT_BYTES = 65536`). Exceeding output triggers an immediate buffer truncation flag. |
| **Environment Sanitization** | Scrubbed Environment Variables | The subprocess inherits **zero ambient credentials, API keys, or workspace paths**. Sensitive environment variables (`API_KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AWS_*`, `GITHUB_*`) are stripped. `PYTHONPATH` is purged. |
| **Process Tree Isolation** | Independent Process Group | Subprocess is spawned with `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` (on Windows) to enable clean termination of child subprocess trees. |
| **AST Pre-Execution Gate** | Static Import & Function Inspection | Disallows high-risk modules (`socket`, `requests`, `urllib`, `winreg`, `ctypes`, `subprocess`) unless explicitly allowlisted for authorized data scripts. |
| **Fail-Closed Default** | `allow_arbitrary_code=False` | Arbitrary user-submitted code is **strictly rejected by default**. Only pre-validated, bounded data manipulation templates can execute without explicit operator authorization. |

---

### 3. Observable Sandbox Results Contract

Every execution returns a structured `SandboxExecutionResult`:
```json
{
  "execution_id": "sbx_1541_001",
  "status": "SUCCESS",
  "exit_code": 0,
  "stdout": "42\n",
  "stderr": "",
  "timed_out": false,
  "output_truncated": false,
  "execution_time_ms": 12.4,
  "workspace_cleaned": true,
  "error": null
}
```
