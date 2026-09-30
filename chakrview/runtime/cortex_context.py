"""
ChakrView Step 56: Cognitive Context Schema & Serializer.

Defines:
- CognitiveContext: Strongly typed dataclass for cortex conditioning.
- serialize_context: Canonical token-bounded XML serialization.
- parse_context: Safe extraction of context fields from prompt strings.
- Enforces 192-token ceiling to preserve ChakrMicro's 512-token context horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import re
from typing import Dict, Any, Optional

MAX_CONTEXT_STRING_CHARS = 800  # Approximates ~192 tokens in Byte-Level BPE


@dataclass
class CognitiveContext:
    task_mode: str = "GENERAL"
    goal: str = ""
    state: str = ""
    memory_context: str = ""
    reasoning_state: str = ""
    constraints: str = ""

    def validate(self) -> None:
        """Enforce length and security bounds."""
        serialized = self.serialize()
        if len(serialized) > MAX_CONTEXT_STRING_CHARS:
            raise ValueError(
                f"CognitiveContext exceeds character budget: {len(serialized)} > {MAX_CONTEXT_STRING_CHARS}"
            )
        # Check prohibited keywords
        prohibited = ("private_key", "secret_key", "BEGIN PRIVATE KEY", "password")
        for p in prohibited:
            if p in serialized:
                raise ValueError(f"Prohibited security token '{p}' found in CognitiveContext.")

    def serialize(self) -> str:
        """Produce canonical XML-tagged cortex context envelope."""
        lines = ["<CORTEX_CONTEXT>"]
        if self.task_mode:
            lines.append(f"<TASK_MODE>{self.task_mode.strip()}</TASK_MODE>")
        if self.goal:
            lines.append(f"<GOAL>{self.goal.strip()}</GOAL>")
        if self.state:
            lines.append(f"<STATE>{self.state.strip()}</STATE>")
        if self.memory_context:
            lines.append(f"<MEMORY>{self.memory_context.strip()}</MEMORY>")
        if self.reasoning_state:
            lines.append(f"<REASONING_STATE>{self.reasoning_state.strip()}</REASONING_STATE>")
        if self.constraints:
            lines.append(f"<CONSTRAINTS>{self.constraints.strip()}</CONSTRAINTS>")
        lines.append("</CORTEX_CONTEXT>\n")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> Optional[CognitiveContext]:
        """Extract CognitiveContext from text if present."""
        match = re.search(r"<CORTEX_CONTEXT>(.*?)</CORTEX_CONTEXT>", text, re.DOTALL)
        if not match:
            return None

        body = match.group(1)

        def _get_tag(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
            return m.group(1).strip() if m else ""

        return cls(
            task_mode=_get_tag("TASK_MODE") or "GENERAL",
            goal=_get_tag("GOAL"),
            state=_get_tag("STATE"),
            memory_context=_get_tag("MEMORY"),
            reasoning_state=_get_tag("REASONING_STATE"),
            constraints=_get_tag("CONSTRAINTS"),
        )
