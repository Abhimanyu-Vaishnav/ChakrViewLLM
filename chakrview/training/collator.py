"""
Batch Collator for ChakrView Pre-Training (Phase 6).

Stacks individual sequence examples into batch tensors:
- input_ids: [B, T]
- target_ids: [B, T]
- attention_mask: [B, T]

Enforces T <= 512 bounds checking and shape consistency without silent truncation.
"""

from typing import List, Dict, Any
import torch


class CausalLanguageModelingCollator:
    """
    Collates individual sequence dictionaries into batched tensors.
    
    Attributes:
        max_context: Maximum allowed context window (frozen at 512).
        pad_token_id: Token ID for padding (2).
    """
    def __init__(self, max_context: int = 512, pad_token_id: int = 2) -> None:
        self.max_context = max_context
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        # Extract items
        input_ids_list = [item["input_ids"] for item in batch]
        target_ids_list = [item["target_ids"] for item in batch]
        attention_mask_list = [
            item.get("attention_mask", torch.ones_like(item["input_ids"]))
            for item in batch
        ]

        # Stack into [B, T]
        input_ids = torch.stack(input_ids_list, dim=0)
        target_ids = torch.stack(target_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        B, T = input_ids.shape

        if T > self.max_context:
            raise ValueError(
                f"Batch sequence length {T} exceeds configured maximum context {self.max_context}. "
                "Silent truncation is prohibited."
            )

        if target_ids.shape != (B, T):
            raise ValueError(
                f"Target shape {target_ids.shape} does not match input shape ({B}, {T})"
            )

        if attention_mask.shape != (B, T):
            raise ValueError(
                f"Attention mask shape {attention_mask.shape} does not match input shape ({B}, {T})"
            )

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }
