# Container Sandbox Security Model & Threat Assessment

## 1. Security Architecture & Threat Vectors

The OCI Container Execution Backend is designed to enforce strict multi-layer boundaries when executing untrusted Python tasks.

```
+---------------------------------------------------------------------------------------+
|                                    HOST MACHINE                                       |
|  +---------------------------------------------------------------------------------+  |
|  |                            ChakrView Governed Router                            |  |
|  |   - Auth & Request Validation    - AST Module Filtering    - Argument Allowlisting  |  |
|  +---------------------------------------------------------------------------------+  |
|                                         |                                             |
|                                 POSIX Exec Array                                      |
|                                         v                                             |
|  +---------------------------------------------------------------------------------+  |
|  |                          OCI Container Sandbox (Docker/Podman)                   |  |
|  |  +---------------------------------------------------------------------------+  |  |
|  |  | Boundary Controls:                                                        |  |  |
|  |  |  * Non-Root User (--user 1000:1000)                                       |  |  |
|  |  |  * Read-Only Root Filesystem (--read-only)                                |  |  |
|  |  |  * Network Disabled (--network none)                                      |  |  |
|  |  |  * Capability Drop (--cap-drop ALL)                                        |  |  |
|  |  |  * Resource Quotas (--memory 256m, --cpus 1.0, --pids-limit 64)            |  |  |
|  |  |  * Zero Host Directory Mounts (Dedicated empty /tmp/workspace only)        |  |  |
|  |  |  * Ephemeral Lifetime (--rm, auto-cleanup on exit/timeout)                  |  |  |
|  |  +---------------------------------------------------------------------------+  |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+
```

---

## 2. Threat Vector Matrix & Security Control Audit

| Threat Vector | Potential Impact | Security Control Enforced | Verification Method | Status on Current Host |
|:---|:---|:---|:---|:---:|
| **Filesystem Escape & Host Tampering** | Arbitrary host modification, repo alteration | `--read-only`, no host mounts, isolated empty workspace | Negative test (write to `/`, attempt to mount host repo) | Enforced in CLI Builder / Simulated |
| **Credential & Secret Leakage** | Exfiltration of API tokens, SSH keys | Scrubbed env vars (`--env-file /dev/null`), no host home directory mounted | Negative test (`os.environ`, read `~/.ssh`) | Enforced in CLI Builder / Simulated |
| **Network Exfiltration / SSRF** | Data theft, scanning internal VPC/metadata services | `--network none` | Negative test (`curl`, `urllib`, socket connect) | Enforced in CLI Builder / Simulated |
| **Denial of Service (CPU Hogging)** | Host unresponsiveness | `--cpus 1.0` and hard process timeout (2.0s ceiling) | Adversarial test (`while True: pass`) | Verified via timeout handler |
| **Denial of Service (OOM Crash)** | Memory exhaustion killing host services | `--memory 256m`, `--memory-swap 256m` | Adversarial test (`[0] * 10**9`) | Enforced in CLI Builder |
| **Process / Fork Bomb** | Host process table exhaustion | `--pids-limit 64` | Adversarial test (`while True: os.fork()`) | Enforced in CLI Builder |
| **Output Flooding** | Memory exhaustion capturing huge stdout | Stream buffer capped at 65,536 bytes (64 KB) | Truncation test (`print('A'*100000)`) | Verified in test suite |
| **Command Injection in Runtime Flags** | Host compromise via shell escape | Argument arrays (`List[str]`), zero shell string interpolation | Adversarial test (injecting `; rm -rf /` in image or args) | Verified in test suite |
| **Privilege Escalation** | Root container escape | `--security-opt=no-new-privileges`, `--cap-drop ALL`, non-root user | Configuration audit | Enforced in CLI Builder |
| **Orphan Container Leakage** | Leaked containers consuming disk and memory | Docker `--rm`, try-finally cleanup logic | Resource audit | Verified in test suite |

---

## 3. Residual Risks & Disclosures
1. **Container Runtime Unavailability:** As audited in `CONTAINER_RUNTIME_AUDIT.md`, unprivileged Windows hosts lack OCI engines; execution safely defaults to disabled (`OCI_RUNTIME_UNAVAILABLE`).
2. **Kernel Exploits:** If the underlying Linux kernel has unpatched container escape CVEs, hypervisor-based containers (e.g. Kata Containers or AWS Firecracker microVMs) provide stronger isolation than shared-kernel OCI containers.
