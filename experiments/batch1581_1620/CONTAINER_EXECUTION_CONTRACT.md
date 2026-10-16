# OCI Container Execution Contract & Interface Specification

## 1. Scope & Interface Architecture

The OCI container backend is encapsulated in `chakrview.runtime.oci_container_backend.OciContainerBackend`. It defines a deterministic, typed contract for executing authorized Python snippets inside an isolated OCI runtime.

---

## 2. Request Data Contract (`ContainerExecutionRequest`)

```python
@dataclass(frozen=True)
class ContainerExecutionRequest:
    task_id: str
    code: str
    image: str = "python:3.11-alpine"
    timeout_seconds: float = 2.0
    memory_limit: str = "256m"
    cpu_limit: str = "1.0"
    pids_limit: int = 64
    network_enabled: bool = False
    read_only_root: bool = True
    custom_env: Dict[str, str] = field(default_factory=dict)
```

### Parameter Validation Rules:
- `task_id`: Alphanumeric identifier `^[a-zA-Z0-9_\-]+$`.
- `image`: Must match an allowlist of pre-approved images. Arbitrary registries or wildcards are rejected.
- `timeout_seconds`: Clamped to `[0.1, 5.0]`. Defaults to `2.0`.
- `memory_limit`: Must match pattern `^\d+[mk]$` with ceiling `512m`.
- `network_enabled`: Strictly defaults to `False`. Must be `False` for untrusted workloads.
- `read_only_root`: Strictly defaults to `True`.
- `custom_env`: Host secrets (`AWS_*`, `TOKEN*`, `KEY*`, `PASSWORD*`) are scrubbed automatically.

---

## 3. Response Data Contract (`ContainerExecutionResponse`)

```python
class ContainerExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    UNSAFE_CONFIG_REJECTED = "UNSAFE_CONFIG_REJECTED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    OVERSIZED_OUTPUT = "OVERSIZED_OUTPUT"

@dataclass
class ContainerExecutionResponse:
    task_id: str
    status: ContainerExecutionStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    execution_time_ms: float
    runtime_name: Optional[str]  # "docker", "podman", or None
    network_disabled: bool
    workspace_cleaned: bool
    error: Optional[str] = None
```

---

## 4. Execution Argument Builder Contract

When an OCI runtime is invoked, the command line array MUST be constructed using the following argument sequence:

```python
cmd = [
    runtime_path, "run", "--rm",
    "--network", "none",
    "--read-only",
    "--user", "1000:1000",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--memory", request.memory_limit,
    "--cpus", request.cpu_limit,
    "--pids-limit", str(request.pids_limit),
    "-v", f"{ephemeral_host_dir}:/workspace:rw",
    "-w", "/workspace",
    request.image,
    "python", "-u", "main.py"
]
```

### Invariants:
1. **No shell string interpolation** (`shell=False` is mandatory).
2. **Dedicated ephemeral host dir only**; no parent path mounts.
3. **Automatic cleanup on exit** via `--rm` plus host-side `tempfile.TemporaryDirectory` context manager.
