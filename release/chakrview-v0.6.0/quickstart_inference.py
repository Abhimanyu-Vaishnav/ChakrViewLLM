"""
release/chakrview-v0.6.0/quickstart_inference.py

Minimal, reproducible standalone inference example for ChakrView v0.6.0.
Loads the canonical Step-10,000 checkpoint and canonical 4096-token BPE tokenizer,
verifies SHA-256 hashes, performs greedy autoregressive text continuation,
and prints structured execution metadata.

Zero training; pure CPU inference.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)


def main():
    print("=" * 65)
    print("CHAKRVIEW v0.6.0 RESEARCH PREVIEW - MINIMAL INFERENCE DEMO")
    print("=" * 65)
    
    ckpt_path, expected_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST[DEFAULT_CHECKPOINT_KEY]
    print(f"Loading Checkpoint: {label}")
    print(f"Path: {ckpt_path.relative_to(REPO_ROOT)}")
    print(f"Expected SHA-256:   {expected_hash[:16]}...{expected_hash[-8:]}")
    
    # Initialize adapter with default greedy decoding
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=ckpt_path,
        max_new_tokens=32,
        decoding="greedy"
    )
    
    # Prompt
    prompt = "ChakrView is an indigenous artificial intelligence"
    print(f"\nInput Prompt: '{prompt}'")
    print("Generating continuation on CPU...")
    
    # Generate
    result = adapter.generate(prompt)
    
    print("\n" + "-" * 65)
    print("GENERATED OUTPUT:")
    print("-" * 65)
    print(f"Prompt Text:     {result['prompt']}")
    print(f"Generated Text:  {result['generated_text']}")
    print(f"Tokens Out:      {result['output_token_count']} tokens")
    print(f"Latency:         {result['latency_ms']:.2f} ms")
    print(f"Weight Verified: {result['weight_hash_verified']}")
    print(f"Compute Device:  {result['device'].upper()}")
    print("-" * 65)
    print("NOTE: ChakrMicro Step 10,000 is a compact (4.5M param) base pretraining model.")
    print("It produces autoregressive text continuations; it is not fine-tuned for conversational instruction following.")
    print("=" * 65)


if __name__ == "__main__":
    main()
