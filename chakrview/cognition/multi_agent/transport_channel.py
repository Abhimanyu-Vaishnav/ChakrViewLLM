"""
ChakrView Step 116: Robust Transport & Node Abstraction.

Establishes transport-neutral message envelopes and abstract worker client/node:
TASK -> ENVELOPE -> TRANSPORT -> WORKER NODE -> RESULT -> VALIDATION -> PPB

Includes:
- BaseTransportChannel (abstract interface for subprocess, localhost, future network)
- SubprocessTransportChannel (production single-node process isolation)
- MessageCorrelation: correlation_id, idempotency_key, payload checksum
- Strict Envelope Serialization & Schema Validation
"""

from __future__ import annotations

import abc
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
    EnvelopeType,
)


@dataclass
class TransportMessageCorrelation:
    """Explicit message provenance tracking."""
    correlation_id: str
    idempotency_key: str
    task_id: str
    worker_id: str
    payload_hash: str
    timestamp: float = field(default_factory=time.time)


class BaseTransportChannel(abc.ABC):
    """
    Abstract interface for transport channels delivering envelopes to worker nodes.
    Supports local child processes now, localhost/network sockets in the future.
    """

    @abc.abstractmethod
    def send_request(
        self,
        request: RequestEnvelope,
        timeout_seconds: float = 10.0,
    ) -> ResponseEnvelope:
        """Sends request envelope and returns response envelope synchronously."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Returns True if the transport destination is reachable."""
        pass


class SubprocessTransportChannel(BaseTransportChannel):
    """
    Concrete transport channel delivering envelopes to isolated OS subprocesses
    via stdin/stdout JSON streaming.
    """

    def __init__(self, runtime_script_path: Path, workspace_root: Path, python_exe: Optional[str] = None) -> None:
        self.runtime_script_path = Path(runtime_script_path).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.python_exe = python_exe or sys.executable

    def is_available(self) -> bool:
        return self.runtime_script_path.exists() and self.workspace_root.exists()

    def send_request(
        self,
        request: RequestEnvelope,
        timeout_seconds: float = 10.0,
    ) -> ResponseEnvelope:
        request.validate()
        cmd = [self.python_exe, str(self.runtime_script_path), str(self.workspace_root)]
        raw_in = json.dumps(request.to_dict())

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_data, stderr_data = proc.communicate(input=raw_in, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise TimeoutError(f"Worker process timed out after {timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Subprocess transport failed to spawn: {str(e)}")

        if proc.returncode != 0:
            raise ProcessLookupError(f"Worker process crashed with exit code {proc.returncode}: {stderr_data.strip()}")

        try:
            resp_dict = json.loads(stdout_data)
            response = ResponseEnvelope.from_dict(resp_dict)
            response.validate()
            return response
        except Exception as e:
            raise ValueError(f"Malformed transport response received: {str(e)}")
