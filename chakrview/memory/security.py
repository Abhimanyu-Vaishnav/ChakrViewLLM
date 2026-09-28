"""
Privacy, Isolation & Security Enforcement for ChakrView Persistent Memory (Step 16).

Enforces core principles:
1. Strict User Isolation: User A's memories NEVER cross into User B's scope.
2. DATA != AUTHORITY: Memory records cannot grant tool permissions or modify system instructions.
3. Secret Redaction: Passwords, tokens, and private credentials are masked before persistence.
"""

from typing import Dict, List, Optional, Any, Set
import re

from chakrview.memory.record import MemoryRecord


class SecurityViolationError(PermissionError):
    """Raised on illegal cross-user memory access or unauthorized execution attempts."""
    pass


class MemorySecurityPolicy:
    """
    Governance policy enforcing tenant isolation and passive data boundaries.
    """

    SENSITIVE_PATTERNS = [
        re.compile(r"(?:api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*([^\s,;]+)", re.IGNORECASE),
        re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),
        re.compile(r"\b([a-zA-Z0-9_-]{32,})\b"),
    ]

    INJECTION_INDICATORS = [
        "system override",
        "ignore previous instructions",
        "execute tool",
        "bypass toolgate",
        "grant admin",
        "delete database",
    ]

    @classmethod
    def sanitize_content(cls, content: str) -> str:
        """
        Redact sensitive tokens, passwords, and API keys from memory text.
        """
        sanitized = content
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = pattern.sub("[REDACTED_CREDENTIAL]", sanitized)
        return sanitized

    @classmethod
    def enforce_owner_isolation(
        cls,
        requesting_owner_id: str,
        records: List[MemoryRecord],
    ) -> List[MemoryRecord]:
        """
        Filter records strictly by requesting_owner_id.
        Raises SecurityViolationError if any record belongs to a different owner.
        """
        for r in records:
            if r.owner_id != requesting_owner_id:
                raise SecurityViolationError(
                    f"Cross-user memory violation: Requesting owner '{requesting_owner_id}' "
                    f"cannot access record '{r.memory_id}' owned by '{r.owner_id}'."
                )
        return [r for r in records if r.owner_id == requesting_owner_id]

    @classmethod
    def assert_no_tool_authority(cls, memory_record: MemoryRecord) -> None:
        """
        Verify that memory records cannot claim tool execution authority.
        DATA != AUTHORITY.
        """
        text = memory_record.content.lower()
        for indicator in cls.INJECTION_INDICATORS:
            if indicator in text:
                # Flag in metadata that memory contains prompt injection attempt
                memory_record.metadata["security_flag"] = "prompt_injection_attempt"
                # Memory remains passive data; authority is never granted
                break
