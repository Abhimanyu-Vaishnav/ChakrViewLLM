"""ChakrView Conversation Alpha Standalone CPU Inference Script.

Authoritative CPU inference engine for ChakrView Conversation Alpha.
Supports multi-turn interaction, greedy decoding, and top-k temperature sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: str = "cpu",
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found at: {tokenizer_dir}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", {})
    cfg = ModelConfig(**raw_config)

    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)
    return model, tokenizer, cfg


def generate_response(
    model: ChakrMicro,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 40,
    device: str = "cpu",
    bos_id: int = 0,
    eos_id: int = 1,
) -> str:
    # Format prompt if missing delimiter
    if not prompt.startswith("User:"):
        prompt = f"User: {prompt}\nAssistant: "
    elif not prompt.endswith("Assistant: ") and not prompt.endswith("Assistant:"):
        prompt = f"{prompt}\nAssistant: "

    prompt_tokens = tokenizer.encode(prompt)
    input_ids = [bos_id] + prompt_tokens
    curr_tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr_tokens.shape[1] >= model.config.max_seq_len:
                break

            logits = model(curr_tokens)
            next_token_logits = logits[0, -1, :]

            if temperature <= 0.0:
                next_token_id = int(torch.argmax(next_token_logits).item())
            else:
                scaled_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[-1]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_token_id == eos_id:
                break

            generated.append(next_token_id)
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            curr_tokens = torch.cat([curr_tokens, next_tensor], dim=1)

    return tokenizer.decode(generated).strip()


def main():
    parser = argparse.ArgumentParser(description="ChakrView Conversation Alpha Inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "chakrview_conversation_alpha.pt",
        help="Path to conversation model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to tokenizer directory containing vocab.json and merges.json",
    )
    parser.add_argument("--prompt", type=str, default="User: Hello! Who are you?\nAssistant: ", help="Prompt string")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling threshold")

    args = parser.parse_args()

    print("=" * 60)
    print("ChakrView Conversation Alpha Standalone Engine")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, cfg = load_model_and_tokenizer(args.checkpoint, args.tokenizer_dir)
    print(f"Architecture: ChakrMicro ({cfg.d_model}d, {cfg.n_layers}L, {cfg.n_heads}H, {cfg.vocab_size}V)")
    print("=" * 60)

    print(f"\n[Prompt]:\n{args.prompt}")
    output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"\n[Generated Output]:\n{output}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
