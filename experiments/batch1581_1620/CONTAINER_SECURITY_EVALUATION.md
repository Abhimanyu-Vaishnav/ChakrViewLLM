# Container Security & Isolation Evaluation Results (Track 1 & Track 2)

## 1. Overview & Denominators
This evaluation reports the security policy enforcement, input validation, resource boundary checks, and fail-closed operational states of the OCI container backend (`chakrview.runtime.oci_container_backend.OciContainerBackend`) and subprocess fallback (`chakrview.runtime.sandboxed_python_tool.SandboxedPythonTool`).

- **Track 1 (Container Policy & Isolation):** 60/60 Passed (**100.0%**)
- **Track 2 (Resource Limits & Recovery):** 60/60 Passed (**100.0%**)

---

## 2. Track 1: Policy Enforcement Results (60 Tasks)

| Category | Task IDs | Tested Security Control | Expected Status | Actual Status | Passed |
|:---|:---:|:---|:---:|:---:|:---:|
| **Runtime Discovery Fail-Closed** | `cont_001` - `cont_020` | Dynamic detection of absent Docker/Podman engine | `RUNTIME_UNAVAILABLE` | `RUNTIME_UNAVAILABLE` | 20/20 |
| **Image Allowlist Filtering** | `cont_021` - `cont_040` | Rejection of unapproved registries/images (`ubuntu`, `busybox`, `evil-corp`) | `UNSAFE_CONFIG_REJECTED` | `UNSAFE_CONFIG_REJECTED` | 20/20 |
| **Unsafe Mount Rejection** | `cont_041` - `cont_050` | Rejection of host tampering paths (`/`, `/etc`, `C:\Windows`, `~/.ssh`) | `UNSAFE_CONFIG_REJECTED` | `UNSAFE_CONFIG_REJECTED` | 10/10 |
| **Network Disabled Assertion** | `cont_051` - `cont_060` | Rejection of requests with `network_enabled=True` | `UNSAFE_CONFIG_REJECTED` | `UNSAFE_CONFIG_REJECTED` | 10/10 |

---

## 3. Track 2: Resource Limits & Failure Recovery (60 Tasks)

| Category | Task IDs | Tested Security Control | Expected Status | Actual Status | Passed |
|:---|:---:|:---|:---:|:---:|:---:|
| **Memory Limit Validation** | `res_061` - `res_080` | Validation of memory argument format and ceiling (`1024g`, `unlimited`) | `UNSAFE_CONFIG_REJECTED` | `UNSAFE_CONFIG_REJECTED` | 20/20 |
| **Timeout Enforcement** | `res_081` - `res_100` | Subprocess wall-clock timeout termination (0.5s limit) | `TIMEOUT` | `TIMEOUT` | 20/20 |
| **Output Buffer Clamping** | `res_101` - `res_120` | Stdout stream buffer truncation at 65,536 bytes (64 KB) | `OVERSIZED_OUTPUT` | `OVERSIZED_OUTPUT` | 20/20 |

---

## 4. Resource Cleanup & Workspace Audit
In all 120 trials across Track 1 and Track 2, host ephemeral directories (`chakr_oci_*` and `chakr_sandbox_*`) were confirmed deleted immediately following process completion or error termination. Zero orphan files or processes persisted on the host.
