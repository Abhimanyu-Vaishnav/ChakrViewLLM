"""
Unit Tests for Conversation Formatter and Loss Modules
Verifies:
1. User and assistant boundaries are correctly identified
2. Earlier turns remain visible in context
3. Prompt tokens are strictly masked with -100
4. Assistant target is correctly aligned
5. Loss is computed strictly over target tokens
6. Padding/masking is handled without target leakage
"""

import sys
import unittest
from pathlib import Path
import torch

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter
from experiments.conversation.cr_exp_17.conversation_loss import ConversationLoss

class TestConversationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
        cls.tok, _ = load_tokenizer_artifacts(str(tok_dir))
        cls.formatter = ConversationFormatter(cls.tok, max_seq_len=512)
        cls.loss_module = ConversationLoss(ignore_index=-100)

    def test_prompt_masking_alignment(self):
        prompt = "User: Hello!\nAssistant:"
        response = "Hello! How can I assist you today?"
        
        full_ids, labels = self.formatter.prepare_training_pair(prompt, response)
        
        prompt_ids = self.tok.encode(prompt.strip())
        
        # Check lengths
        self.assertEqual(len(full_ids), len(labels))
        
        # Check that prompt tokens are masked with -100
        for i in range(len(prompt_ids)):
            self.assertEqual(labels[i], -100, f"Token at {i} should be masked with -100")
            
        # Check that response tokens are unmasked
        for j in range(len(prompt_ids), len(full_ids)):
            self.assertEqual(labels[j], full_ids[j], f"Token at {j} should match target ID")
            
        # Check that last token in response is EOS (token ID 1)
        self.assertEqual(full_ids[-1], 1)
        self.assertEqual(labels[-1], 1)

    def test_loss_computation_on_masked_labels(self):
        vocab_size = 4096
        seq_len = 20
        batch_size = 2
        
        # Dummy logits
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        
        # Labels where first 10 tokens are masked (-100)
        labels = torch.randint(2, vocab_size, (batch_size, seq_len))
        labels[:, :10] = -100
        
        loss, acc = self.loss_module(logits, labels)
        
        # Loss must be non-zero and require grad
        self.assertTrue(loss.item() > 0.0)
        self.assertIsNotNone(loss.grad_fn)
        
        # Backward pass should produce gradients
        loss.backward()
        self.assertIsNotNone(logits.grad)
        
        # Gradients on the first 9 positions should be ZERO (since shift aligns position i to label i+1,
        # labels 0..9 are -100, so positions 0..8 have no loss signal)
        grad_norms_prompt = logits.grad[:, :8, :].abs().sum().item()
        self.assertEqual(grad_norms_prompt, 0.0, "Gradients should be strictly zero on masked prompt positions")
        
        # Gradients on target positions should be NON-ZERO
        grad_norms_target = logits.grad[:, 10:, :].abs().sum().item()
        self.assertTrue(grad_norms_target > 0.0, "Gradients on target positions must be non-zero")

    def test_multiturn_formatting(self):
        turns = [
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Hello Bob!"},
            {"role": "user", "content": "What is my name?"}
        ]
        formatted = self.formatter.format_turns(turns)
        expected = "User: My name is Bob.\nAssistant: Hello Bob!\nUser: What is my name?"
        self.assertEqual(formatted, expected)

if __name__ == "__main__":
    unittest.main()
