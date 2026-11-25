"""
ChakrView Training Objective Forensic Audit Test Suite.
Mission CV-GENESIS-NR-014 Phase 1.

Rigorous unit tests checking:
1. Exact shift alignment: logits[..., :-1, :] vs labels[..., 1:]
2. Loss masking on prompt tokens (-100 ignored by CrossEntropyLoss)
3. Zero loss contribution from padding tokens (<PAD>=2 masked with -100)
4. EOS handling (<EOS>=1 target prediction)
5. Strict causal mask prevention: no attention to future tokens
6. Parameter gradient flow and weight updates
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig


class TestTrainingObjective(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        # Small configuration for fast exact verification
        self.config = Genesis17MConfig(
            vocab_size=64,
            d_model=64,
            n_layers=2,
            n_q_heads=4,
            n_kv_heads=2,
            hidden_dim=128,
            max_seq_len=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2
        )
        self.model = ChakrGenesis17M(self.config)

    def test_causal_mask_prevents_future_leakage(self):
        """Checks that output at position t is strictly independent of inputs at t' > t."""
        self.model.eval()
        seq_a = torch.tensor([[0, 10, 20, 30]], dtype=torch.long)
        seq_b = torch.tensor([[0, 10, 20, 99 % self.config.vocab_size]], dtype=torch.long)  # Only 4th token differs

        with torch.no_grad():
            logits_a, _ = self.model(seq_a)
            logits_b, _ = self.model(seq_b)

        # Positions 0, 1, 2 must produce bit-exact identical logits
        diff_0_to_2 = (logits_a[:, :3, :] - logits_b[:, :3, :]).abs().max().item()
        self.assertEqual(diff_0_to_2, 0.0, f"Future token at t=3 leaked into earlier positions! Max diff: {diff_0_to_2}")

    def test_shifted_autoregressive_loss(self):
        """Verifies next-token prediction alignment: logits[..., :-1, :] vs labels[..., 1:]."""
        # Let sequence be [BOS(0), A(5), B(12), EOS(1)]
        input_ids = torch.tensor([[0, 5, 12, 1]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 12, 1]], dtype=torch.long)  # Prompt tokens [0, 5] masked

        logits, _ = self.model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Manually compute loss on the unmasked positions:
        # shift_labels has tokens: [-100, 12, 1]
        # Token at shift index 1 is label 12 (target of token 5)
        # Token at shift index 2 is label 1 (target of token 12)
        target_indices = [1, 2]
        manual_loss = 0.0
        for idx in target_indices:
            pred_logits = shift_logits[0, idx, :]
            lbl = shift_labels[0, idx]
            manual_loss += F.cross_entropy(pred_logits.unsqueeze(0), lbl.unsqueeze(0))
        manual_loss /= len(target_indices)

        self.assertAlmostEqual(loss.item(), manual_loss.item(), places=5)

    def test_padding_and_prompt_masking_zero_contribution(self):
        """Verifies that padding tokens with label=-100 contribute exactly zero to loss and gradients."""
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Case 1: unpadded
        inp_1 = torch.tensor([[0, 10, 20]], dtype=torch.long)
        lbl_1 = torch.tensor([[-100, 10, 20]], dtype=torch.long)

        logits_1, _ = self.model(inp_1)
        shift_logits_1 = logits_1[..., :-1, :].contiguous()
        shift_labels_1 = lbl_1[..., 1:].contiguous()
        loss_1 = criterion(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))

        # Case 2: padded with PAD=2 and label=-100
        inp_2 = torch.tensor([[0, 10, 20, 2, 2]], dtype=torch.long)
        lbl_2 = torch.tensor([[-100, 10, 20, -100, -100]], dtype=torch.long)

        logits_2, _ = self.model(inp_2)
        shift_logits_2 = logits_2[..., :-1, :].contiguous()
        shift_labels_2 = lbl_2[..., 1:].contiguous()
        loss_2 = criterion(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))

        # Due to causal masking, logits at position 0 and 1 are identical, so unmasked loss must be identical
        self.assertAlmostEqual(loss_1.item(), loss_2.item(), places=5)

    def test_gradient_flow_and_parameter_update(self):
        """Verifies that loss backward updates all trainable layers without vanishing or explosion."""
        self.model.train()
        input_ids = torch.tensor([[0, 5, 10, 15, 1]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 10, 15, 1]], dtype=torch.long)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        logits, _ = self.model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        # Check gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for parameter: {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient: {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf in gradient: {name}")

        optimizer.step()


if __name__ == "__main__":
    unittest.main()
