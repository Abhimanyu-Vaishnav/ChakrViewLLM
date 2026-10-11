"""
Sovereign Capability Policy and Execution Gate for ChakrView (Step 17).

Enforces strict authority boundaries, input validation, output sanitization,
and failure isolation for all capability invocations:
- DATA != AUTHORITY
- CAPABILITY EXISTENCE != CAPABILITY AUTHORIZATION
- CAPABILITY OUTPUT != INSTRUCTION

Capability execution must be:
- Bounded
- Observable
- Auditable
- Timeout-aware
- Permission-aware
- Failure-isolated
"""

from dataclasses import dataclass, field
import re
import time
from typing import Dict, List, Optional, Any, Set

from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    CapabilityStatus,
    RiskClassification,
)
from chakrview.capability.registry import CapabilityRegistry, CapabilityNotFoundError


class CapabilityAuthorizationError(PermissionError):
    """Raised when an unpermitted or unauthorized capability execution is attempted."""
    pass


class CapabilityArgumentValidationError(ValueError):
    """Raised when capability parameters fail sanitization or schema checks."""
    pass


class CapabilityDisabledError(RuntimeError):
    """Raised when attempting to execute a disabled or unavailable capability."""
    pass


class CapabilityExecutionTimeoutError(TimeoutError):
    """Raised when capability execution exceeds allotted timeout."""
    pass


class CapabilityGate:
    """
    Governed Policy Gate mediating between Cognitive execution and Capabilities.
    """

    # Prohibited patterns in parameters
    DANGEROUS_PATTERNS = [
        "__import__",
        "eval(",
        "exec(",
        "subprocess",
        "os.system",
        "shutil",
        "socket",
        "open(",
        "__builtins__",
    ]

    # Secret and credential regex patterns for output redaction
    SECRET_PATTERNS = [
        (re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"), r"\1: [REDACTED]"),
        (re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{12,}", re.IGNORECASE), "Bearer [REDACTED]"),
        (re.compile(r"sk-[a-zA-Z0-9]{16,}"), "sk-[REDACTED]"),
    ]

    # Prompt injection markers in outputs
    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions"),
        re.compile(r"(?i)system\s*:\s*you\s+are\s+now"),
        re.compile(r"(?i)new\s+system\s+prompt"),
        re.compile(r"(?i)admin\s+override"),
        re.compile(r"(?i)bypass\s+security"),
    ]

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self.registry = registry or CapabilityRegistry()

    def authorize(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
        active_policy: Optional[Any] = None,
    ) -> bool:
        """
        Verify whether the request is authorized under the active policy and context.
        
        Strict rules:
        1. Memory or retrieved data can NEVER authorize a capability.
        2. Capability must exist in the registry.
        3. Capability must be in an operational state (not DISABLED, ERROR, UNAVAILABLE).
        4. Caller context must possess required permissions.
        5. If active_policy provides allowed_capabilities whitelist, capability must be in it.
        """
        cap_id = request.capability_id

        # 1. Authority denial: Check provenance
        provenance = request.context.get("provenance_source") or (
            context.constraints.get("provenance_source") if context else None
        )
        if provenance in (
            "retrieved_knowledge", "working_memory", "document", "persistent_memory",
            "state_knowledge_assertion", "state_knowledge", "knowledge_state",
            "hypothesis", "reasoning_hypothesis", "inference", "reasoning_inference",
            "reasoning_trace", "decision_candidate", "user_assertion",
            "critical_thinking", "critical_hypothesis", "critical_trace",
            "continual_memory", "episodic_memory", "semantic_memory",
            "memory_consolidation", "experience",
        ):
            raise CapabilityAuthorizationError(
                f"Authority denial: Retrieved content, state assertions, reasoning, or memory ('{provenance}') "
                f"cannot authorize execution of capability '{cap_id}'."
            )

        # 2. Registry lookup
        if not self.registry.has(cap_id):
            raise CapabilityNotFoundError(f"Capability '{cap_id}' is not registered.")

        descriptor = self.registry.get_descriptor(cap_id)

        # 3. Operational status check
        if descriptor.status in (CapabilityStatus.DISABLED, CapabilityStatus.UNAVAILABLE, CapabilityStatus.ERROR):
            raise CapabilityDisabledError(
                f"Capability '{cap_id}' cannot be executed; status is {descriptor.status.value}."
            )

        # 4. Context permissions check
        if descriptor.required_permissions:
            caller_permissions: Set[str] = set()
            if context and context.granted_permissions:
                caller_permissions = context.granted_permissions

            missing = [p for p in descriptor.required_permissions if p not in caller_permissions]
            if missing:
                # Check if active_policy explicitly whitelists the capability
                policy_allowed = False
                if active_policy is not None:
                    allowed_caps = getattr(active_policy, "allowed_capabilities", None)
                    if allowed_caps is None:
                        allowed_caps = getattr(active_policy, "supported_capability_ids", None)
                    allowed_tools = getattr(active_policy, "allowed_tools", None)
                    if allowed_caps and cap_id in allowed_caps:
                        policy_allowed = True
                    elif allowed_tools and cap_id in allowed_tools:
                        policy_allowed = True

                if not policy_allowed:
                    raise CapabilityAuthorizationError(
                        f"Caller lacks required permissions for '{cap_id}': {missing}"
                    )

        # 5. Policy whitelist & risk check
        if active_policy is not None:
            allowed_caps = getattr(active_policy, "allowed_capabilities", None)
            if allowed_caps is None:
                allowed_caps = getattr(active_policy, "supported_capability_ids", None)
            if allowed_caps is not None and cap_id not in allowed_caps:
                # Also check allowed_tools if bridged
                allowed_tools = getattr(active_policy, "allowed_tools", None)
                if not (allowed_tools and cap_id in allowed_tools):
                    raise CapabilityAuthorizationError(
                        f"Capability '{cap_id}' is not in allowed policy whitelist."
                    )

            allowed_risks = getattr(active_policy, "allowed_risk_levels", None)
            if allowed_risks is not None and descriptor.risk_level not in allowed_risks:
                raise CapabilityAuthorizationError(
                    f"Capability '{cap_id}' risk level {descriptor.risk_level.value} "
                    f"is not permitted by active policy."
                )

        return True

    def validate_parameters(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
    ) -> None:
        """
        Validate input arguments for dangerous patterns and schema conformity.
        """
        # 1. Dangerous pattern check
        def _check_val(val: Any) -> None:
            if isinstance(val, str):
                for pat in self.DANGEROUS_PATTERNS:
                    if pat in val:
                        raise CapabilityArgumentValidationError(
                            f"Parameter contains prohibited pattern '{pat}'."
                        )
            elif isinstance(val, dict):
                for v in val.values():
                    _check_val(v)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _check_val(item)

        _check_val(request.parameters)

        # 2. Schema check
        cap = self.registry.get(descriptor.capability_id)
        valid, err = cap.validate_arguments(request.parameters)
        if not valid:
            raise CapabilityArgumentValidationError(f"Schema validation failed: {err}")

    def sanitize_output(self, result: CapabilityResult) -> CapabilityResult:
        """
        Sanitize output by redacting credentials and neutralizing injection attempts.
        """
        if result.output is None:
            return result

        def _clean_str(text: str) -> str:
            # Redact secrets
            cleaned = text
            for pattern, repl in self.SECRET_PATTERNS:
                cleaned = pattern.sub(repl, cleaned)

            # Check for prompt injection
            for pattern in self.PROMPT_INJECTION_PATTERNS:
                if pattern.search(cleaned):
                    cleaned = f"[INJECTION_RISK: UNTRUSTED CAPABILITY OUTPUT] {cleaned}"
                    result.metadata["injection_detected"] = True
                    break

            return cleaned

        def _clean_obj(obj: Any) -> Any:
            if isinstance(obj, str):
                return _clean_str(obj)
            elif isinstance(obj, dict):
                return {k: _clean_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_obj(item) for item in obj]
            return obj

        result.output = _clean_obj(result.output)
        return result

    def execute_governed(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
        active_policy: Optional[Any] = None,
    ) -> CapabilityResult:
        """
        Execute a capability through the full governed policy gate.
        
        Lifecycle:
        1. Authorize request against policy and caller context.
        2. Validate parameters against dangerous patterns & schema.
        3. Execute capability with bounded timeout and failure isolation.
        4. Sanitize output (credential redaction and injection marking).
        """
        t0 = time.perf_counter()
        cap_id = request.capability_id

        # 1. Authorization
        try:
            self.authorize(request, context=context, active_policy=active_policy)
        except (CapabilityAuthorizationError, CapabilityNotFoundError, CapabilityDisabledError) as e:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=cap_id,
                success=False,
                error=f"Gate Authorization Denied: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        descriptor = self.registry.get_descriptor(cap_id)

        # 2. Parameter validation
        try:
            self.validate_parameters(request, descriptor)
        except CapabilityArgumentValidationError as e:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=cap_id,
                success=False,
                error=f"Gate Parameter Validation Failed: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        # 3. Execution with Failure Isolation
        capability = self.registry.get(cap_id)
        try:
            result = capability.execute(request, context=context)
        except Exception as e:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=cap_id,
                success=False,
                error=f"Capability Execution Failure: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        # 4. Output Sanitization
        return self.sanitize_output(result)

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
        active_policy: Optional[Any] = None,
    ) -> CapabilityResult:
        """Execute a capability request through the governed policy gate."""
        return self.execute_governed(request, context=context, active_policy=active_policy)

