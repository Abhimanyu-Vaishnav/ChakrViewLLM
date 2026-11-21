"""
ChakrView Conversation CLI Interface: experiments/ucip29/chat_cli.py
Provides a clean, interactive local conversation interface for ChakrConv v0.1.
Supports:
- Multi-turn sessions with conversation history accumulation.
- Configurable generation parameters (greedy, temperature, repetition penalty, max tokens).
- Session reset (/reset), info display (/info), and clean exit (/exit).
- Pure neural generation (Zero hidden deterministic shortcuts).
- 100% CPU inference.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings
from experiments.ucip29.architecture.chakrconv import ChakrConv

DEFAULT_CHECKPOINT = ROOT / "releases" / "ucip29_candidate" / "chakrconv_v01_best.pt"


def compute_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ChakrChatSession:
    def __init__(self, checkpoint_path: Path, max_history_turns: int = 5):
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.exists():
            alt = ROOT / "experiments" / "ucip29" / "training" / "checkpoint_best.pt"
            if alt.exists():
                self.checkpoint_path = alt
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.sha256 = compute_sha256(self.checkpoint_path)
        print(f"Loading ChakrConv neural model from: {self.checkpoint_path.name}")
        print(f"Verified Checkpoint SHA-256: {self.sha256}")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ModelConfig(**ckpt["config"])
        self.model = ChakrConv(self.config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
        self.history: List[Tuple[str, str]] = []
        self.max_history_turns = max_history_turns

    def reset(self) -> None:
        self.history.clear()
        print("\n[Conversation history reset to empty]\n")

    def format_prompt(self, user_message: str) -> str:
        prompt = ""
        recent_history = self.history[-self.max_history_turns:]
        for u, a in recent_history:
            prompt += f"User: {u}\nAssistant: {a}\n"
        prompt += f"User: {user_message}\nAssistant: "
        return prompt

    def generate_reply(
        self,
        user_message: str,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15
    ) -> Tuple[str, float]:
        full_prompt = self.format_prompt(user_message)
        p_str, _ = normalize_prompt_response_strings(full_prompt, "", strip_delimiters=True)
        input_ids = [0] + self.tokenizer.encode(p_str)
        x = torch.tensor([input_ids], dtype=torch.long)

        t0 = time.time()
        with torch.no_grad():
            gen_tokens = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        elapsed = time.time() - t0

        reply = self.tokenizer.decode(gen_tokens, errors="replace").strip()
        self.history.append((user_message, reply))
        return reply, elapsed


def interactive_cli():
    parser = argparse.ArgumentParser(description="ChakrView Local Conversational Chat CLI")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty")
    args = parser.parse_args()

    session = ChakrChatSession(Path(args.checkpoint))

    print("\n" + "=" * 60)
    print("  ChakrView Conversation Model v0.1-alpha (CPU-First)")
    print("  Commands: /reset (clear history) | /info | /exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input == "/exit":
                print("Exiting session. Goodbye!")
                break
            elif user_input == "/reset":
                session.reset()
                continue
            elif user_input == "/info":
                print(f"Model: ChakrConv (d_model={session.config.d_model}, layers={session.config.n_layers})")
                print(f"Checkpoint SHA-256: {session.sha256}")
                print(f"History turns: {len(session.history)}")
                continue

            reply, latency = session.generate_reply(
                user_input,
                temperature=args.temp,
                repetition_penalty=args.rep_penalty
            )
            print(f"Assistant > {reply}  ({latency * 1000.0:.1f}ms)\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


if __name__ == "__main__":
    interactive_cli()
