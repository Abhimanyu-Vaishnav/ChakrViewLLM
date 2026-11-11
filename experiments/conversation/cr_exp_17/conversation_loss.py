"""
Phase D: Conversation Loss Module (conversation_loss.py)
Provides assistant-target cross-entropy loss computation
ignoring prompt tokens masked with -100.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ConversationLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes next-token prediction loss shifted by 1 position:
        logits[:, :-1, :] predicts labels[:, 1:]
        Returns (loss, active_token_accuracy)
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        vocab_size = shift_logits.size(-1)
        loss = self.loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # Calculate accuracy strictly over non-ignored target tokens
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            mask = (shift_labels != -100)
            if mask.sum() > 0:
                acc = ((preds == shift_labels) & mask).sum().float() / mask.sum().float()
            else:
                acc = torch.tensor(0.0, device=logits.device)
                
        return loss, acc
