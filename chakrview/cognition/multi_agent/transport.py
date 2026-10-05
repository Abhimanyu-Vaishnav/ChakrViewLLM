"""
ChakrView Step 114: Transport-Neutral Envelope and Protocol for Process-Isolated Workers.

Defines schemas for process-isolated requests and responses:
- RequestEnvelope
- ResponseEnvelope
- Protocol validation and envelope hashing
- Transport abstraction interface
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract, WorkerPackage
from chakrview.cognition.multi_agent.result import WorkerResult, WorkerExecutionStatus


PROTOCOL_VERSION_V1 = "1.0.0"


class EnvelopeType(str, Enum):
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESPONSE = "TASK_RESPONSE"
    HEARTBEAT = "HEARTBEAT"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class RequestEnvelope:
    """
    Explicit wire envelope carrying a bounded task execution order to an isolated process.
    """
    protocol_version: str
    worker_id: str
    task_id: str
    package_id: str
    role: str
    context_hash: str
    context_token_count: int
    allowed_tools: List[str]
    allowed_files: List[str]
    context_budget: int
    package_payload: Dict[str, Any]  # Serialized WorkerPackage
    expected_output_schema: Dict[str, Any] = field(default_factory=dict)
    envelope_type: EnvelopeType = EnvelopeType.TASK_REQUEST
    timestamp: float = field(default_factory=time.time)

    def compute_hash(self) -> str:
        data = f"{self.protocol_version}:{self.worker_id}:{self.task_id}:{self.package_id}:{self.context_hash}:{self.context_token_count}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RequestEnvelope:
        data_copy = dict(data)
        if "envelope_type" in data_copy and isinstance(data_copy["envelope_type"], str):
            data_copy["envelope_type"] = EnvelopeType(data_copy["envelope_type"])
        return cls(**data_copy)

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION_V1:
            raise ValueError(f"Incompatible protocol_version: {self.protocol_version}")
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("Invalid envelope: worker_id must not be empty")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("Invalid envelope: task_id must not be empty")
        if self.context_token_count <= 0 or self.context_token_count > 512:
            raise ValueError(f"Context ceiling violation: {self.context_token_count} exceeds 512 tokens")
        if self.context_budget <= 0 or self.context_budget > 512:
            raise ValueError(f"Context budget ceiling violation: {self.context_budget} exceeds 512 tokens")


@dataclass
class ResponseEnvelope:
    """
    Explicit wire envelope carrying execution results back from an isolated worker process.
    """
    protocol_version: str
    worker_id: str
    task_id: str
    package_id: str
    result_status: str  # WorkerExecutionStatus value
    result_payload: Dict[str, Any]  # Serialized WorkerResult
    result_hash: str
    error: Optional[str] = None
    envelope_type: EnvelopeType = EnvelopeType.TASK_RESPONSE
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResponseEnvelope:
        data_copy = dict(data)
        if "envelope_type" in data_copy and isinstance(data_copy["envelope_type"], str):
            data_copy["envelope_type"] = EnvelopeType(data_copy["envelope_type"])
        return cls(**data_copy)

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION_V1:
            raise ValueError(f"Incompatible protocol_version: {self.protocol_version}")
        if not self.worker_id or not self.task_id:
            raise ValueError("Invalid response envelope: missing worker_id or task_id")
        if not self.result_status:
            raise ValueError("Invalid response envelope: missing result_status")
        # Validate result hash matches payload
        payload_bytes = json.dumps(self.result_payload, sort_keys=True).encode("utf-8")
        computed = hashlib.sha256(payload_bytes).hexdigest()
        if self.result_hash != computed:
            raise ValueError(f"Result hash mismatch! Expected {computed}, got {self.result_hash}")
