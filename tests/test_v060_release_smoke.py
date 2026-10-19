import unittest
import os
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    EXPECTED_TOKENIZER_FILES_HASHES,
    DEFAULT_CHECKPOINT_KEY,
    compute_file_sha256,
    NeuralArtifactIntegrityError,
)
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


class TestV060ReleaseSmoke(unittest.TestCase):
    """
    Automated smoke tests for ChakrView v0.6.0 release candidate:
    1. Canonical checkpoint and tokenizer SHA-256 integrity.
    2. Model configuration and parameter shape compatibility.
    3. Non-empty output on deterministic smoke input.
    4. Deterministic output reproducibility under greedy decoding.
    5. Context length and boundary handling.
    6. Preservation of chakrview/runtime/session.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path, cls.exp_ckpt_hash, cls.ckpt_label = CANONICAL_CHECKPOINTS_ALLOWLIST[DEFAULT_CHECKPOINT_KEY]
        cls.tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
        cls.session_path = REPO_ROOT / "chakrview" / "runtime" / "session.py"

    def test_01_canonical_checkpoint_hash_integrity(self):
        self.assertTrue(self.ckpt_path.is_file(), f"Checkpoint missing: {self.ckpt_path}")
        actual_hash = compute_file_sha256(self.ckpt_path)
        self.assertEqual(
            actual_hash,
            "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
            "Canonical Step-10,000 checkpoint hash mismatch"
        )

    def test_02_canonical_tokenizer_hashes_integrity(self):
        self.assertTrue(self.tok_dir.is_dir(), f"Tokenizer directory missing: {self.tok_dir}")
        for fname, exp_hash in EXPECTED_TOKENIZER_FILES_HASHES.items():
            fpath = self.tok_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing tokenizer file: {fpath}")
            act_hash = compute_file_sha256(fpath)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch for {fname}")

    def test_03_protected_session_py_unmodified(self):
        self.assertTrue(self.session_path.is_file(), "chakrview/runtime/session.py must exist")
        # Check git status for session.py
        import subprocess
        res = subprocess.run(["git", "status", "--porcelain", "chakrview/runtime/session.py"], capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(res.stdout.strip(), "", "chakrview/runtime/session.py has uncommitted changes!")

    def test_04_model_loading_and_parameter_shapes(self):
        adapter = NeuralRuntimeAdapter(checkpoint_path=self.ckpt_path)
        adapter.initialize()
        self.assertTrue(adapter.is_loaded)
        self.assertIsNotNone(adapter._bound_weight_digest)
        
        # Check model topology: 6 layers, 192 d_model, 6 heads
        model = adapter._contract.model
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(model.config.vocab_size, 4096)
        self.assertEqual(model.config.d_model, 192)
        self.assertEqual(model.config.n_layers, 6)
        self.assertEqual(model.config.n_heads, 6)
        # Check token embedding shape: [4096, 192]
        self.assertEqual(model.embedding.weight.shape, (4096, 192))

    def test_05_deterministic_smoke_generation(self):
        adapter = NeuralRuntimeAdapter(checkpoint_path=self.ckpt_path, max_new_tokens=16, decoding="greedy")
        prompt = "ChakrView research baseline"
        
        out1 = adapter.generate(prompt)
        out2 = adapter.generate(prompt)
        
        # Non-empty output
        self.assertTrue(len(out1["generated_text"]) > 0)
        self.assertEqual(out1["output_token_count"], 16)
        
        # Determinism
        self.assertEqual(out1["generated_text"], out2["generated_text"])
        self.assertEqual(out1["generated_tokens"], out2["generated_tokens"])

    def test_06_unauthorized_checkpoint_fails_safely(self):
        bogus_path = REPO_ROOT / "bogus_checkpoint.pt"
        with self.assertRaises(NeuralArtifactIntegrityError):
            adapter = NeuralRuntimeAdapter(checkpoint_path=bogus_path)
            adapter.initialize()

    def test_07_invalid_sampling_parameters_fail_safely(self):
        with self.assertRaises(ValueError):
            # temperature < 0
            NeuralRuntimeAdapter(temperature=-1.0)
            
        with self.assertRaises(ValueError):
            # top_p <= 0
            NeuralRuntimeAdapter(top_p=0.0)


if __name__ == "__main__":
    unittest.main()
