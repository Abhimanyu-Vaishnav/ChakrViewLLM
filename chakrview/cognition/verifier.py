"""
Independent Verification Layer for ChakrView Cognitive Subsystem (Step 15).

Validates structured step outputs, types, schemas, and assertions independently
from language model generation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union

from chakrview.cognition.planner import PlanStep
from chakrview.cognition.observation import StepObservation


@dataclass
class VerificationResult:
    """
    Outcome of an independent verification check.

    Attributes:
        passed: True if all verification assertions succeeded.
        notes: Human-readable diagnostic description of outcome.
        details: Specific check results and violation details.
    """
    passed: bool
    notes: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(**data)


class StepVerifier:
    """
    Deterministic step output verifier.
    
    Checks:
    - Non-empty constraints
    - Type conformity ('number', 'str', 'dict', 'list', 'bool')
    - Key presence in dictionary results
    - Numeric range boundaries (min_value, max_value)
    - Custom predicate functions
    """

    @staticmethod
    def verify(
        step: PlanStep,
        observation: StepObservation,
        custom_validators: Optional[Dict[str, Callable[[Any], Tuple[bool, str]]]] = None,
    ) -> VerificationResult:
        """
        Verify an observation against the step's verification_requirements.
        """
        if not observation.success:
            return VerificationResult(
                passed=False,
                notes=f"Step execution failed: {observation.error or 'Unknown error'}",
                details={"step_id": step.step_id, "error": observation.error},
            )

        reqs = step.verification_requirements
        if not reqs:
            # No verification constraints specified -> passes by default
            return VerificationResult(
                passed=True,
                notes="No verification constraints defined; output accepted.",
                details={"step_id": step.step_id},
            )

        output = observation.output
        violations: List[str] = []
        details: Dict[str, Any] = {"step_id": step.step_id}

        # 1. Non-empty check
        allow_empty = reqs.get("allow_empty", True)
        if not allow_empty:
            if output is None or (isinstance(output, (str, list, dict, set, tuple)) and len(output) == 0):
                violations.append("Output must not be empty.")

        # 2. Expected type check
        expected_type = reqs.get("expected_type")
        if expected_type and output is not None:
            if expected_type == "number":
                if not isinstance(output, (int, float)) or isinstance(output, bool):
                    violations.append(f"Expected numeric output, got {type(output).__name__}")
            elif expected_type == "str":
                if not isinstance(output, str):
                    violations.append(f"Expected string output, got {type(output).__name__}")
            elif expected_type == "dict":
                if not isinstance(output, dict):
                    violations.append(f"Expected dict output, got {type(output).__name__}")
            elif expected_type == "list":
                if not isinstance(output, list):
                    violations.append(f"Expected list output, got {type(output).__name__}")
            elif expected_type == "bool":
                if not isinstance(output, bool):
                    violations.append(f"Expected boolean output, got {type(output).__name__}")

        # 3. Required dictionary keys
        required_keys = reqs.get("required_keys")
        if required_keys and isinstance(output, dict):
            for k in required_keys:
                if k not in output:
                    violations.append(f"Missing required key '{k}' in output dict.")

        # 4. Numeric range checks
        if isinstance(output, (int, float)) and not isinstance(output, bool):
            min_val = reqs.get("min_value")
            if min_val is not None and output < min_val:
                violations.append(f"Value {output} is below minimum allowed {min_val}.")
            max_val = reqs.get("max_value")
            if max_val is not None and output > max_val:
                violations.append(f"Value {output} is above maximum allowed {max_val}.")

        # 5. Custom validator
        custom_rule = reqs.get("custom_rule")
        if custom_rule and custom_validators and custom_rule in custom_validators:
            func = custom_validators[custom_rule]
            custom_pass, custom_note = func(output)
            if not custom_pass:
                violations.append(f"Custom validation '{custom_rule}' failed: {custom_note}")

        if violations:
            details["violations"] = violations
            return VerificationResult(
                passed=False,
                notes="; ".join(violations),
                details=details,
            )

        return VerificationResult(
            passed=True,
            notes="All verification checks passed successfully.",
            details=details,
        )
