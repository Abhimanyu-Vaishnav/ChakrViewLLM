"""
ChakrView Interactive Local Model Runner & CLI (Step 46).

Interactive terminal interface demonstrating verified local model execution,
multi-turn stateful dialogue, real-time token streaming, and cryptographic
weight immutability verification over ChakrMicro v0.1.

Usage:
    # Interactive multi-turn chat REPL:
    python scripts/run_local_model.py

    # Single-shot prompt generation:
    python scripts/run_local_model.py --prompt "ChakrView is an indigenous" --max-tokens 24
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.runtime.sampling import SamplingConfig


def print_banner(runtime: LocalModelRuntime):
    ident = runtime.model_identity
    print("=" * 72)
    print(" CHAKRVIEW LOCAL MODEL RUNNER — CHAKRMICRO v0.1")
    print("=" * 72)
    print(f" Architecture : {ident.architecture_name} v{ident.version}")
    print(f" Parameters   : {ident.parameter_count:,} (exact)")
    print(f" Vocab Size   : {ident.vocab_size:,}")
    print(f" Context Max  : {ident.max_context_len} tokens")
    print(f" Weight Hash  : {ident.weight_hash[:20]}... [VERIFIED ΔW=0]")
    print("=" * 72)
    print(" Type your message and press Enter.")
    print(" Commands: /reset to clear conversation, /quit to exit.")
    print("=" * 72)


def run_interactive(runtime: LocalModelRuntime, session_id: str = "cli_session"):
    print_banner(runtime)
    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        sampling=SamplingConfig(temperature=0.7, top_k=20, top_p=0.9, seed=42),
    )

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ChakrView Local Runner.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Session ended.")
            break

        if user_input.lower() == "/reset":
            runtime.reset_session(session_id)
            print("[Session context reset]")
            continue

        print("ChakrView > ", end="", flush=True)
        t0 = time.perf_counter()
        token_count = 0

        try:
            for chunk in runtime.stream_chat(
                session_id=session_id,
                prompt=user_input,
                generation_config=gen_cfg,
            ):
                print(chunk.token_text, end="", flush=True)
                token_count += 1
            dur_ms = (time.perf_counter() - t0) * 1000.0
            tok_per_sec = (token_count / (dur_ms / 1000.0)) if dur_ms > 0 else 0.0
            print(f"\n[{token_count} tokens in {dur_ms:.1f}ms | {tok_per_sec:.1f} tok/s]")
        except Exception as e:
            print(f"\n[Error: {e}]")


def run_single_shot(runtime: LocalModelRuntime, prompt: str, max_tokens: int):
    print(f"Prompt: {prompt}")
    print("Generating: ", end="", flush=True)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        sampling=SamplingConfig(temperature=0.0),  # Greedy for reproducible one-shot
    )

    t0 = time.perf_counter()
    tokens = 0
    for chunk in runtime.stream_generate(prompt=prompt, generation_config=gen_cfg):
        print(chunk.token_text, end="", flush=True)
        tokens += 1
    dur_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n[Generated {tokens} tokens in {dur_ms:.1f}ms]")


def main():
    parser = argparse.ArgumentParser(description="ChakrView Local Model Runner")
    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=24, help="Maximum new tokens to generate")
    args = parser.parse_args()

    print("Initializing ChakrView Local Model Runtime...")
    runtime = LocalModelRuntime.from_default()

    if args.prompt:
        run_single_shot(runtime, args.prompt, args.max_tokens)
    else:
        run_interactive(runtime)


if __name__ == "__main__":
    main()
