"""
ChakrView Interactive Local Model Evaluation CLI (Step 50).

Provides a terminal chat session allowing direct interaction with the best trained
ChakrView checkpoint, alongside side-by-side comparison with the frozen baseline,
structured probe evaluation, raw probability inspection, and quantitative metric tracking.

Run command:
    .venv\\Scripts\\python.exe scripts/chat_chakrview.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reconfigure console streams to handle Unicode on Windows safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from chakrview.runtime.interactive import (
    InteractiveModelSessionCoordinator,
    DEFAULT_TRAINED_CHECKPOINT,
    DEFAULT_TOKENIZER_DIR,
    EXPECTED_WEIGHT_HASH,
)


def print_banner(coord: InteractiveModelSessionCoordinator) -> None:
    info = coord.get_info()
    print("==================================================")
    print("              CHAKRVIEW LOCAL MODEL               ")
    print("==================================================")
    print()
    print(f"Checkpoint:     {info['checkpoint']}")
    print(f"Training Steps: {info['training_steps']}")
    print(f"Parameters:     {info['parameters']:,}")
    print(f"Tokenizer:      {info['tokenizer_checksum'][:16]}...")
    print(f"Context:        {info['context_length']}")
    print()
    print("Type /help for commands.")
    print("Type /quit to exit.")
    print("==================================================")
    print()


def print_help() -> None:
    print("\nAvailable Commands:")
    print("  /help             Show this help message")
    print("  /info             Display model architecture, checkpoint, and weight digests")
    print("  /reset            Reset conversation context history")
    print("  /context          Display current turn count and remaining token budget")
    print("  /stats            Display session metrics (tokens, latency, throughput, repetition)")
    print("  /compare          Run prompt on BOTH trained model and frozen baseline")
    print("  /probe            Execute structured evaluation probe suite (Probes A-E)")
    print("  /raw              Toggle raw token-level diagnostics (probabilities, rank, top-5)")
    print("  /seed <N|random>  Set RNG seed (e.g. /seed 42) or randomize (/seed random)")
    print("  /generate_config  Display or inspect active generation hyperparameters")
    print("  /save [path]      Save structured session log to artifacts/step50/sessions/")
    print("  /quit             Exit interactive session safely\n")


def handle_compare(coord: InteractiveModelSessionCoordinator, user_prompt: Optional[str] = None) -> None:
    if not user_prompt:
        try:
            user_prompt = input("Enter prompt to compare: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

    if not user_prompt:
        print("Empty prompt. Comparison skipped.")
        return

    print("\nRunning side-by-side comparison on identical parameters...")
    res = coord.compare_prompt(user_prompt)

    print()
    print("PROMPT:")
    print(res.prompt)
    print()
    print("BASELINE:")
    print(res.baseline_text if res.baseline_text else "<empty>")
    print()
    print("TRAINED:")
    print(res.trained_text if res.trained_text else "<empty>")
    print()
    print("METRICS")
    print("----------------------------")
    print(f"Baseline tokens:     {res.baseline_metrics.length}")
    print(f"Trained tokens:      {res.trained_metrics.length}")
    print()
    print(f"Baseline repetition: {res.baseline_metrics.repetition_unigram:.4f}")
    print(f"Trained repetition:  {res.trained_metrics.repetition_unigram:.4f}")
    print()
    print(f"Baseline EOS:        {res.baseline_metrics.eos_occurred}")
    print(f"Trained EOS:         {res.trained_metrics.eos_occurred}")
    print()
    print(f"Baseline latency:    {res.baseline_metrics.elapsed_sec * 1000.0:.2f} ms")
    print(f"Trained latency:     {res.trained_metrics.elapsed_sec * 1000.0:.2f} ms")
    print("----------------------------\n")


def handle_probe(coord: InteractiveModelSessionCoordinator) -> None:
    print("\nExecuting Structured Probe Suite from tests/fixtures/step50_interactive_probes.json...")
    t0 = time.perf_counter()
    probes_res = coord.run_probes()
    elapsed = time.perf_counter() - t0

    probes = probes_res.get("probes", {})

    print(f"\n--- PROBE A: BASIC LANGUAGE CONTINUATIONS ---")
    for item in probes.get("probe_a", []):
        print(f"Prompt: \"{item['prompt']}\"")
        print(f"  Top-1: \"{item['top1_token']}\" (p={item['top1_probability']:.4f})")
        print(f"  Continuation: \"{item['continuation']}\"")

    print(f"\n--- PROBE B: CONTEXT-SENSITIVITY DIVERGENCE ---")
    for item in probes.get("probe_b", []):
        print(f"Pair: \"{item['prompt_1']}\" vs \"{item['prompt_2']}\"")
        print(f"  JS Divergence: {item['js_divergence']:.4f} | Cosine Dist: {item['cosine_distance']:.4f} | Top-5 Overlap: {item['top5_overlap']:.2f}")

    print(f"\n--- PROBE C: CONTROLLED MEMORY / DISTANCE SENSITIVITY ---")
    for item in probes.get("probe_c", []):
        print(f"Distance {item['distance_tokens']} tok ({item['target_1']} vs {item['target_2']}): JS={item['js_divergence']:.4f} | Cos={item['cosine_distance']:.4f}")

    print(f"\n--- PROBE D: NEGATIVE CONTEXT CONTROL ---")
    for item in probes.get("probe_d", []):
        print(f"Clean vs Corrupted: JS Divergence = {item['js_divergence']:.4f} | Cosine Dist = {item['cosine_distance']:.4f}")

    print(f"\n--- PROBE E: SIMPLE PATTERN COMPLETION ---")
    for item in probes.get("probe_e", []):
        top1 = item["top5"][0] if item["top5"] else {}
        print(f"Pattern \"{item['pattern']}\" -> Top-1: \"{top1.get('token_str', '')}\" (p={top1.get('probability', 0):.4f})")

    print(f"\nProbe execution completed in {elapsed:.2f}s.\n")


def run_interactive(coord: InteractiveModelSessionCoordinator, scripted_commands: Optional[List[str]] = None) -> None:
    print_banner(coord)

    command_iter = iter(scripted_commands) if scripted_commands is not None else None

    while True:
        try:
            if command_iter is not None:
                try:
                    user_input = next(command_iter)
                    print(f"You: {user_input}")
                except StopIteration:
                    break
            else:
                user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        # Command Dispatch
        cmd = user_input.split()[0].lower()
        args = user_input.split()[1:]

        if cmd in ("/quit", "/exit"):
            save_path = coord.save_session_log()
            print(f"Session saved to {save_path}")
            print("Goodbye.")
            break

        elif cmd == "/help":
            print_help()

        elif cmd == "/info":
            info = coord.get_info()
            print("\nMODEL & SESSION INFO:")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print()

        elif cmd == "/reset":
            coord.reset_context()
            print("Conversation context reset.\n")

        elif cmd == "/context":
            stats = coord.get_context_stats()
            print("\nCONTEXT STATUS:")
            print(f"  Active Turns:             {stats['turns']}")
            print(f"  Tokens in Context:        {stats['token_count']}")
            print(f"  Remaining Context Budget: {stats['remaining_context_budget']} tokens\n")

        elif cmd == "/stats":
            stats = coord.get_session_stats()
            print("\nSESSION STATISTICS:")
            print(f"  Prompts Processed:        {stats['prompts']}")
            print(f"  Generated Tokens:         {stats['generated_tokens']}")
            print(f"  Average Latency:          {stats['average_latency_ms']:.2f} ms")
            print(f"  Average Throughput:       {stats['average_tokens_per_sec']:.2f} tokens/sec")
            print(f"  EOS Count:                {stats['eos_count']}")
            print(f"  Unigram Repetition Ratio: {stats['repetition_ratio']:.4f}\n")

        elif cmd == "/raw":
            active = coord.toggle_raw()
            state_str = "ENABLED" if active else "DISABLED"
            print(f"Raw token diagnostics: {state_str}\n")

        elif cmd == "/seed":
            if args:
                val = args[0]
                s = coord.set_seed(val)
                if s is not None:
                    print(f"RNG seed set to {s} (deterministic mode).\n")
                else:
                    print("RNG randomized (non-deterministic mode).\n")
            else:
                print(f"Current seed: {coord.seed} (deterministic={coord.deterministic})\n")

        elif cmd == "/generate_config":
            print("\nACTIVE GENERATION CONFIG:")
            cfg = coord.gen_config.to_dict()
            for k, v in cfg.items():
                print(f"  {k}: {v}")
            print()

        elif cmd == "/compare":
            prompt_arg = " ".join(args) if args else None
            handle_compare(coord, prompt_arg)

        elif cmd == "/probe":
            handle_probe(coord)

        elif cmd == "/save":
            save_arg = args[0] if args else None
            path = coord.save_session_log(save_arg)
            print(f"Session log saved to: {path}\n")

        elif cmd.startswith("/"):
            print(f"Unknown command: '{cmd}'. Type /help for available commands.\n")

        else:
            # Normal conversational turn
            response_text, metrics, raw_diags = coord.generate_turn(user_input)

            if coord.raw_diagnostics and raw_diags:
                print("\n--- RAW TOKEN DIAGNOSTICS ---")
                for diag in raw_diags[:10]:  # Show first 10 for clarity
                    print(f"Step {diag.step:2d} | Token: \"{diag.token_text}\" (ID: {diag.token_id}) | p={diag.probability:.4f} | Rank: {diag.rank}")
                    top5_str = ", ".join([f'"{alt["token_str"]}": {alt["probability"]:.4f}' for alt in diag.top5_alternatives])
                    print(f"        Top-5: {top5_str}")
                if len(raw_diags) > 10:
                    print(f"... [{len(raw_diags) - 10} more tokens generated]")
                print("-----------------------------\n")

            print(f"ChakrView: {response_text if response_text else '<empty>'}")
            print(f"[{metrics.length} tokens, {metrics.latency_ms_per_token:.1f} ms/tok, {metrics.throughput_tokens_per_sec:.1f} tok/s]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChakrView Interactive Model Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_TRAINED_CHECKPOINT),
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=str(DEFAULT_TOKENIZER_DIR),
        help="Directory containing tokenizer artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic evaluation (default 42)",
    )
    args = parser.parse_args()

    coordinator = InteractiveModelSessionCoordinator(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=Path(args.tokenizer_dir),
        deterministic=True,
        seed=args.seed,
    )

    run_interactive(coordinator)


if __name__ == "__main__":
    main()
