"""
Unit and regression test suite for SFT Prompt Conditioning Alignment.
Location: tests/test_sft_prompt_alignment.py
"""

import json
import unittest
from pathlib import Path
import tempfile
import numpy as np

from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset

class TestSFTPromptAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.tok, _ = load_tokenizer_artifacts(str(cls.repo_root / "data" / "experiments" / "vocab_4096"))

    def test_sft_dataset_uses_formatted_prompt_when_available(self):
        """Verifies that SFTDataset encodes formatted_prompt directly when present."""
        records = [
            {
                "id": "ALIGN_01",
                "prompt": "What is the capital of France?",
                "formatted_prompt": "Instruction: What is the capital of France?\nResponse:",
                "response": "Paris is the capital of France.",
                "prompt_mode": "plain_instruction"
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)

        try:
            ds = SFTDataset(
                jsonl_path=tmp_path,
                tokenizer=self.tok,
                max_seq_len=512,
                allow_exploratory_opt_in=True,
                enforce_trust_gate=False,
                is_training=True,
            )
            rng = np.random.RandomState(42)
            x, y, active_count = ds.get_batch(batch_size=1, rng=rng)

            # Check prompt tokens match format_conditioned_prompt
            expected_prompt_str = "Instruction: What is the capital of France?\nResponse:"
            expected_prompt_toks = self.tok.encode(expected_prompt_str)

            # Target sequence should mask exactly len(expected_prompt_toks) tokens with -100
            prompt_masked_count = (y[0] == -100).sum().item()
            self.assertEqual(prompt_masked_count, len(expected_prompt_toks))

            # Response tokens should be unmasked
            expected_resp_toks = self.tok.encode("Paris is the capital of France.")
            self.assertEqual(active_count, len(expected_resp_toks) + 1) # +1 for EOS

            # The input sequence should start with BOS (0), followed by prompt tokens, then response tokens
            input_list = x[0].tolist()
            self.assertEqual(input_list[0], 0) # BOS
            self.assertEqual(input_list[1:1+len(expected_prompt_toks)], expected_prompt_toks)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_sft_dataset_formats_prompt_dynamically_when_formatted_prompt_missing(self):
        """Verifies that if formatted_prompt is omitted, SFTDataset uses prompt_mode and context."""
        records = [
            {
                "id": "ALIGN_02",
                "prompt": "Who is the lead technician?",
                "context": "Facility log: Alex is the lead technician on duty.",
                "response": "Alex is the lead technician.",
                "prompt_mode": "context_qa"
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            tmp_path = Path(f.name)

        try:
            ds = SFTDataset(
                jsonl_path=tmp_path,
                tokenizer=self.tok,
                max_seq_len=512,
                allow_exploratory_opt_in=True,
                enforce_trust_gate=False,
                is_training=True,
            )
            rng = np.random.RandomState(42)
            x, y, active_count = ds.get_batch(batch_size=1, rng=rng)

            expected_prompt_str = format_conditioned_prompt(
                "Who is the lead technician?",
                prompt_mode="context_qa",
                context="Facility log: Alex is the lead technician on duty."
            )
            expected_prompt_toks = self.tok.encode(expected_prompt_str)

            prompt_masked_count = (y[0] == -100).sum().item()
            self.assertEqual(prompt_masked_count, len(expected_prompt_toks))

            input_list = x[0].tolist()
            self.assertEqual(input_list[1:1+len(expected_prompt_toks)], expected_prompt_toks)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_token_and_mask_parity_with_runtime_inference(self):
        """Validates that training input prompt tokens exactly match NeuralRuntimeAdapter prompt tokens."""
        from chakrview.runtime.neural_inference_contract import NeuralInferenceContract
        from chakrview.brain.config import ModelConfig
        import torch

        prompt = "What is the speed of sound?"
        mode = "plain_instruction"
        formatted_prompt = format_conditioned_prompt(prompt, prompt_mode=mode)

        # Inference tokenization path:
        contract_tokens = self.tok.encode(formatted_prompt) # encode via tokenizer

        # Training dataset tokenization path:
        records = [{"id": "PARITY_01", "prompt": prompt, "response": "Sound travels at 343 m/s.", "prompt_mode": mode}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(records[0]) + "\n")
            tmp_path = Path(f.name)

        try:
            ds = SFTDataset(tmp_path, self.tok, max_seq_len=512, enforce_trust_gate=False, is_training=True)
            x, y, _ = ds.get_batch(batch_size=1, rng=np.random.RandomState(42))

            # The prompt tokens in training input must be identical to inference tokens
            training_prompt_slice = x[0, 1:1+len(contract_tokens)].tolist()
            self.assertEqual(training_prompt_slice, contract_tokens)

            # All training prompt tokens must be masked with -100 in target
            training_target_slice = y[0, :len(contract_tokens)].tolist()
            self.assertEqual(training_target_slice, [-100] * len(contract_tokens))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

if __name__ == "__main__":
    unittest.main()
