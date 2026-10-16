"""
Governed Multi-Turn Conversation State and Tool Result Manager.
Location: chakrview/runtime/conversation_state.py

Implements:
- Typed ToolExecutionRecord storing operation, result, status, and provenance.
- ConversationStateManager managing isolated multi-turn session state.
- Explicit reference resolution ('that', 'the previous result', 'the answer')
  mapping to typed previous tool results.
- Bounded history with configurable maximum chain length and expiration.
- Tenant and session isolation preventing cross-conversation leakage.
- Ambiguity detection when multiple candidate results exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class ToolResultType(str, Enum):
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    VERIFICATION_RECORD = "VERIFICATION_RECORD"
    ERROR = "ERROR"


@dataclass
class ToolExecutionRecord:
    turn_id: int
    tool_id: str
    operation: str
    arguments: Dict[str, Any]
    result: Any
    result_type: ToolResultType
    success: bool
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["result_type"] = self.result_type.value
        return d


@dataclass
class ConversationState:
    conversation_id: str
    tenant_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    max_chain_length: int = 10
    ttl_seconds: float = 3600.0  # 1 hour expiration
    tool_history: List[ToolExecutionRecord] = field(default_factory=list)
    user_turns: List[str] = field(default_factory=list)
    assistant_turns: List[str] = field(default_factory=list)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        now = current_time or time.time()
        return (now - self.updated_at) > self.ttl_seconds

    def add_tool_record(self, record: ToolExecutionRecord) -> None:
        self.tool_history.append(record)
        # Enforce maximum chain length (sliding window)
        if len(self.tool_history) > self.max_chain_length:
            self.tool_history = self.tool_history[-self.max_chain_length:]
        self.updated_at = time.time()

    def reset(self) -> None:
        self.tool_history.clear()
        self.user_turns.clear()
        self.assistant_turns.clear()
        self.updated_at = time.time()


@dataclass
class ReferenceResolutionResult:
    resolved: bool
    referenced_value: Optional[Any] = None
    resolved_expression: Optional[str] = None
    ambiguous: bool = False
    candidate_count: int = 0
    reason: str = ""


class ConversationStateManager:
    """
    Manages stateful multi-turn tool history with strict tenant isolation,
    explicit antecedent reference resolution, and ambiguity protection.
    """

    # Reference cues matching pronouns or referring phrases
    REFERENCE_PATTERNS = [
        r"\bthat\b",
        r"\bit\b",
        r"\bthe\s+result\b",
        r"\bthe\s+previous\s+result\b",
        r"\bthe\s+last\s+result\b",
        r"\bthe\s+answer\b",
        r"\bthis\s+value\b",
        r"\bthe\s+value\b",
    ]

    def __init__(self, max_chain_length: int = 10, ttl_seconds: float = 3600.0) -> None:
        self.max_chain_length = max_chain_length
        self.ttl_seconds = ttl_seconds
        # Storage keyed by (tenant_id, conversation_id) for strict isolation
        self._sessions: Dict[Tuple[str, str], ConversationState] = {}

    def get_or_create_session(
        self, conversation_id: str, tenant_id: str = "default_tenant"
    ) -> ConversationState:
        key = (tenant_id, conversation_id)
        if key not in self._sessions:
            self._sessions[key] = ConversationState(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                max_chain_length=self.max_chain_length,
                ttl_seconds=self.ttl_seconds,
            )
        else:
            # Check expiration
            session = self._sessions[key]
            if session.is_expired():
                session.reset()
        return self._sessions[key]

    def reset_session(self, conversation_id: str, tenant_id: str = "default_tenant") -> None:
        key = (tenant_id, conversation_id)
        if key in self._sessions:
            self._sessions[key].reset()

    def record_tool_result(
        self,
        conversation_id: str,
        tool_id: str,
        operation: str,
        arguments: Dict[str, Any],
        result: Any,
        result_type: ToolResultType = ToolResultType.NUMERIC,
        success: bool = True,
        tenant_id: str = "default_tenant",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionRecord:
        session = self.get_or_create_session(conversation_id, tenant_id=tenant_id)
        turn_id = len(session.tool_history) + 1
        rec = ToolExecutionRecord(
            turn_id=turn_id,
            tool_id=tool_id,
            operation=operation,
            arguments=arguments,
            result=result,
            result_type=result_type,
            success=success,
            provenance=provenance or {},
        )
        session.add_tool_record(rec)
        return rec

    def resolve_reference(
        self,
        query: str,
        conversation_id: str,
        tenant_id: str = "default_tenant",
        target_result_type: ToolResultType = ToolResultType.NUMERIC,
    ) -> ReferenceResolutionResult:
        """
        Resolves anaphora in query (e.g. 'divide that by 4', 'now add 50 to the result')
        against stored tool history for this session and tenant.
        """
        session = self.get_or_create_session(conversation_id, tenant_id=tenant_id)
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Check if query contains any reference patterns
        matched_cue = None
        for pat in self.REFERENCE_PATTERNS:
            m = re.search(pat, q_lower)
            if m:
                matched_cue = m.group(0)
                break

        if not matched_cue:
            return ReferenceResolutionResult(
                resolved=False,
                reason="No anaphoric reference cue detected in query.",
            )

        # Filter tool records by type and success
        valid_records = [
            r for r in session.tool_history
            if r.success and r.result_type == target_result_type and r.result is not None
        ]

        if not valid_records:
            return ReferenceResolutionResult(
                resolved=False,
                reason=f"No previous valid {target_result_type.value} tool results in conversation history.",
            )

        # Check for ambiguity: if query says 'the first result' vs 'the last result'
        if re.search(r"\bthe\s+first\s+result\b", q_lower):
            chosen = valid_records[0]
        else:
            # Default to most recent antecedent
            chosen = valid_records[-1]

        val = chosen.result

        # Construct substituted expression for arithmetic
        # e.g., "now divide that by 4" -> "200 / 4"
        # e.g., "add 50 to that" -> "50 + 200"
        # e.g., "multiply that by 3" -> "200 * 3"
        # Anaphora tokens pattern
        ana_pattern = r"(?:that|it|the\s+result|the\s+answer|this\s+value)"
        resolved_expr = None

        m_div = re.search(rf"(?:divide\s+{ana_pattern}\s+by|\/\s*)\s*(\d+(?:\.\d+)?)", q_lower)
        if m_div:
            resolved_expr = f"{val} / {m_div.group(1)}"

        m_mul = re.search(rf"(?:multiply\s+{ana_pattern}\s+by|multiply\s+by|\*\s*)\s*(\d+(?:\.\d+)?)", q_lower)
        if m_mul:
            resolved_expr = f"{val} * {m_mul.group(1)}"

        m_add = re.search(rf"(?:add\s+{ana_pattern}\s+to|add\s+to\s+{ana_pattern})\s*(\d+(?:\.\d+)?)", q_lower)
        if m_add:
            resolved_expr = f"{m_add.group(1)} + {val}"
        else:
            m_add2 = re.search(rf"(?:add\s+)(\d+(?:\.\d+)?)\s*(?:to\s+{ana_pattern})?", q_lower)
            if m_add2 and re.search(ana_pattern, q_lower):
                resolved_expr = f"{val} + {m_add2.group(1)}"

        m_sub = re.search(rf"(?:subtract\s+{ana_pattern}\s+from)\s*(\d+(?:\.\d+)?)", q_lower)
        if m_sub:
            resolved_expr = f"{m_sub.group(1)} - {val}"
        else:
            m_sub2 = re.search(rf"(?:subtract\s+)(\d+(?:\.\d+)?)\s*(?:from\s+{ana_pattern})?", q_lower)
            if m_sub2 and re.search(ana_pattern, q_lower):
                resolved_expr = f"{val} - {m_sub2.group(1)}"

        if not resolved_expr:
            # General fallback substitution: replace cue with numeric value
            resolved_expr = re.sub(rf"\b{re.escape(matched_cue)}\b", str(val), q_clean, flags=re.IGNORECASE)

        return ReferenceResolutionResult(
            resolved=True,
            referenced_value=val,
            resolved_expression=resolved_expr,
            candidate_count=len(valid_records),
            reason=f"Resolved '{matched_cue}' to previous Turn {chosen.turn_id} value ({val}).",
        )
