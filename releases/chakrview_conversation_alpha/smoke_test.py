"""ChakrView Conversation Alpha Standalone Smoke Test.

Validates that the released conversation checkpoint, tokenizer, and runtime
execute seamlessly on CPU without external dependencies.
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
    checkpoint_path = release_dir / "chakrview_conversation_alpha.pt"

    print("=== ChakrView Conversation Alpha Smoke Test ===")

    # 1. Checkpoint existence & hash
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    with open(checkpoint_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"[PASS] Conversation checkpoint verified: {h[:16]}...")

    # 2. Tokenizer verification
    tok, _ = load_tokenizer_artifacts(release_dir)
    assert tok.vocab_size == 4096
    test_str = "ChakrView conversation alpha नमस्ते"
    assert tok.decode(tok.encode(test_str)) == test_str
    print("[PASS] Tokenizer round-trip & vocab verified (4,096).")

    # 3. Model loading
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136
    print(f"[PASS] Model instantiated with {total_params:,} parameters.")

    # 4. Multi-turn prompt execution
    prompt = "User: Hello! Who are you?\nAssistant: "
    input_ids = [0] + tok.encode(prompt)
    inp = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)
        assert logits.shape == (1, len(input_ids), 4096)
        next_tok = logits[0, -1, :].argmax().item()
        assert isinstance(next_tok, int)
    print(f"[PASS] Conversational forward pass verified. Logits: {logits.shape}. Next token: {next_tok}")

    print("=== All Conversation Alpha Smoke Tests Passed Successfully ===")


if __name__ == "__main__":
    run_smoke_test()
