"""ChakrView Alpha v0.1 Release Smoke Test.

Verifies end-to-end CPU execution, checkpoint parameter integrity,
tokenizer loading, and generative forward passes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_smoke_test():
    release_dir = Path(__file__).resolve().parent
    checkpoint_path = ROOT / "experiments" / "ucip13" / "checkpoints" / "primary_seed42" / "checkpoint_step_800.pt"

    print("=== Running ChakrView Alpha v0.1 Smoke Test ===")
    
    # 1. Checkpoint existence and integrity
    assert checkpoint_path.exists(), f"Checkpoint missing: {checkpoint_path}"
    with open(checkpoint_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
    assert ckpt_hash == expected_hash, f"Checkpoint hash mismatch: got {ckpt_hash}"
    print(f"[PASS] Base Checkpoint SHA-256 verified: {ckpt_hash[:16]}...")

    # 2. Tokenizer loading
    tok, _ = load_tokenizer_artifacts(release_dir)
    assert tok.vocab_size == 4096, f"Vocab size mismatch: {tok.vocab_size}"
    test_str = "ChakrView indigenous intelligence नमस्ते"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    assert decoded == test_str, f"Round-trip failed: '{decoded}' != '{test_str}'"
    print("[PASS] Tokenizer round-trip and vocabulary size (4,096) verified.")

    # 3. Model construction and state loading
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Parameter count mismatch: {total_params} != 3443136"
    print(f"[PASS] Model instantiated with {total_params:,} parameters (ChakrMicro v0.1).")

    # 4. Generative forward pass
    tokens = [0] + tok.encode("User: Hello\nAssistant:")
    inp = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(inp)
        assert out.shape == (1, len(tokens), 4096), f"Output shape unexpected: {out.shape}"
        next_tok = out[0, -1].argmax().item()
        assert isinstance(next_tok, int)
    print(f"[PASS] Generative forward pass successful. Logits: {out.shape}. Next token: {next_tok}")

    print("=== All Release Smoke Tests Passed Successfully ===")


if __name__ == "__main__":
    run_smoke_test()
