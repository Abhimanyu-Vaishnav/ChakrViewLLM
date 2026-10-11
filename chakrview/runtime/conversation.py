"""
Conversation Template and Multi-Turn SFT Representation for ChakrView.
Supports system, user, and assistant turns with explicit delimiters:
  <|im_start|>system\n...<|im_end|>\n
  <|im_start|>user\n...<|im_end|>\n
  <|im_start|>assistant\n...<|im_end|>\n

Encodable via the standard frozen 4096-vocabulary BPE tokenizer without inventing
out-of-vocabulary special token IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatMessage:
        role_str = data["role"].lower()
        return cls(
            role=Role(role_str),
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class ConversationTemplate:
    """
    Standard delimiter-based template serializer and parser.
    Format per turn:
      <|im_start|>{role}\n{content}<|im_end|>\n
    """

    START_TAG = "<|im_start|>"
    END_TAG = "<|im_end|>"

    @classmethod
    def format_message(cls, role: Role | str, content: str) -> str:
        role_name = role.value if isinstance(role, Role) else role
        return f"{cls.START_TAG}{role_name}\n{content.strip()}{cls.END_TAG}\n"

    @classmethod
    def format_conversation(cls, messages: List[ChatMessage], add_generation_prompt: bool = False) -> str:
        """
        Renders a full conversation into a formatted dialogue string.
        """
        output_parts = []
        for msg in messages:
            output_parts.append(cls.format_message(msg.role, msg.content))

        if add_generation_prompt:
            output_parts.append(f"{cls.START_TAG}assistant\n")

        return "".join(output_parts)

    @classmethod
    def parse_conversation(cls, formatted_text: str) -> List[ChatMessage]:
        """
        Parses formatted dialogue text back into structured ChatMessage objects.
        """
        messages = []
        parts = formatted_text.split(cls.START_TAG)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if cls.END_TAG in part:
                turn_body = part.split(cls.END_TAG)[0]
            else:
                turn_body = part

            lines = turn_body.split("\n", 1)
            role_header = lines[0].strip().lower()
            content = lines[1].strip() if len(lines) > 1 else ""

            if role_header in [r.value for r in Role]:
                messages.append(ChatMessage(role=Role(role_header), content=content))

        return messages

    @classmethod
    def encode_multiturn_sft(
        cls,
        messages: List[ChatMessage],
        tokenizer: Any,
        max_seq_len: int = 512,
    ) -> Tuple[List[int], List[int], int]:
        """
        Builds causal input tokens and target tokens where system and user turns
        are masked with -100, and assistant turns are scored for next-token prediction.
        """
        bos_id = 0
        eos_id = 1
        ignore_index = -100

        input_tokens: List[int] = [bos_id]
        target_tokens: List[int] = []

        for msg in messages:
            if msg.role == Role.ASSISTANT:
                prefix = f"{cls.START_TAG}assistant\n"
                prefix_ids = tokenizer.encode(prefix)
                content_ids = tokenizer.encode(msg.content.strip())
                suffix = f"{cls.END_TAG}\n"
                suffix_ids = tokenizer.encode(suffix)

                turn_input = prefix_ids + content_ids + suffix_ids
                # In causal shift:
                # The prefix is input context -> target is masked (-100)
                # The content + suffix are assistant response -> active targets
                turn_target = ([ignore_index] * len(prefix_ids)) + content_ids + suffix_ids

                input_tokens.extend(turn_input)
                target_tokens.extend(turn_target)
            else:
                # SYSTEM or USER turn: completely masked from loss
                turn_str = cls.format_message(msg.role, msg.content)
                turn_ids = tokenizer.encode(turn_str)
                input_tokens.extend(turn_ids)
                target_tokens.extend([ignore_index] * len(turn_ids))

        # Append terminal EOS
        input_tokens = input_tokens[:max_seq_len]
        # Align targets: target predicts input_tokens[i+1]
        # Input has leading BOS (position 0), target has len(input_tokens)-1 predictions
        # To align: target_tokens corresponds to inputs[1:] + [EOS]
        aligned_targets = target_tokens[: len(input_tokens) - 1] + [eos_id]

        active_tokens = sum(1 for t in aligned_targets if t != ignore_index)
        return input_tokens, aligned_targets, active_tokens
