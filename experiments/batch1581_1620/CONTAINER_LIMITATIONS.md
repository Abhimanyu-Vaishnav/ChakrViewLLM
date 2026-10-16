# Container Limitations & Operational Boundaries

## 1. Operating System & Platform Boundaries
- **Primary Host OS:** Windows 11 / Windows Server (unprivileged user context).
- **OCI Container Runtime Availability:** `docker`, `podman`, and `nerdctl` are **not installed** or accessible.
- **WSL2 / Linux Virtualization:** Windows Subsystem for Linux is not installed on this host.
- **Operational Reality:** On the current Windows host, true OCI container isolation cannot be executed. The system therefore operates in an explicit **fail-closed disabled state** (`OCI_RUNTIME_UNAVAILABLE`), rejecting arbitrary code execution and documenting exact requirements.

---

## 2. Distinction Between Isolation Tiers

| Isolation Tier | Underlying Mechanism | Network Isolation | Filesystem Isolation | Resource Quotas | Security Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Tier 1: OCI Container Sandbox** | Linux namespaces, cgroups v2, seccomp, rootless user | Enforced (`--network none`) | Enforced (`--read-only`, empty temp workspace) | Hard kernel cgroups (CPU, RAM, PID limits) | **Production-Grade Sandbox** (when Docker/Podman present) |
| **Tier 2: Host Subprocess Fallback** | Process execution with AST static filter & sanitized env | Application-level (blocked `socket` AST import) | Ephemeral directory cleanup (not kernel-isolated) | Subprocess timeout (2.0s), buffer capping (64 KB) | **Restricted Dev Fallback ONLY** (Not an OCI sandbox) |
| **Tier 3: Standard Subprocess** | Raw `subprocess.Popen` | None | None | Wall-clock timeout only | **INSECURE FOR UNTRUSTED CODE** |

> **Critical Rule:** ChakrView never claims Tier 2 is Tier 1. If Tier 1 is absent, arbitrary execution remains disabled by default.

---

## 3. Disallowed Configurations & Fail-Closed Invariants
1. **No Privileged Containers:** Flags like `--privileged`, `--cap-add`, or mounting `/dev` are strictly forbidden.
2. **No Host Filesystem Mounts:** Any mount request targeting the ChakrView repository, user home directory (`~`), or system directories (`/etc`, `C:\Windows`) is rejected with `UNSAFE_MOUNT_REJECTED`.
3. **No Dynamic Image Pulls:** Image names must match an explicit allowlist (e.g. `docker.io/library/python:3.11-alpine`). Arbitrary external image URLs are rejected.
4. **No Shell String Invocation:** Runtime CLI arguments are strictly constructed as string arrays (`argv: List[str]`) to prevent shell injection.
