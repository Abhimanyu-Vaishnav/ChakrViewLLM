"""
ChakrView Interactive CLI Application (chakrview-chat).
Mission CV-GENESIS-NR-012.

Features:
- Multi-turn interactive conversational dialogue directly in terminal.
- Mode switching: /mode neural | /mode hybrid.
- Working memory inspection: /memory.
- Working memory reset: /clear.
- Session export to JSON: /export <filepath>.
- Real-time generation latency and token throughput stats.
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.runtime.conversation_session import ConversationSession


def run_cli():
    parser = argparse.ArgumentParser(description="ChakrView Interactive Terminal Chat (CV-GENESIS-NR-012)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt",
        help="Path to 17M model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/experiments/vocab_4096",
        help="Path to tokenizer directory or vocab.json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["pure_neural", "hybrid"],
        help="Default dialogue mode (pure_neural or hybrid)",
    )
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per response")

    args = parser.parse_args()

    print("=" * 60)
    print("  CHAKRVIEW INTERACTIVE CONVERSATIONAL RUNTIME")
    print("  Mission: CV-GENESIS-NR-012 | Model: ChakrGenesis-17M")
    print("=" * 60)
    print(f"Loading checkpoint: {args.checkpoint}")
    engine = ChakrViewInferenceEngine17M(args.checkpoint, args.vocab)
    print(f"Model loaded successfully on device: {engine.device.type.upper()}")
    print("Commands:")
    print("  /mode <neural|hybrid> : Toggle dialogue mode")
    print("  /memory               : Inspect working memory and facts")
    print("  /clear                : Reset session and erase working memory")
    print("  /export <file.json>   : Export conversation log")
    print("  /quit or exit         : Exit application")
    print("-" * 60)

    session = ConversationSession(engine, mode=args.mode)
    print(f"Session started: {session.session_id} | Active Mode: [{session.mode.upper()}]\n")

    while True:
        try:
            user_input = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ChakrView Chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit", "exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ["neural", "pure_neural"]:
                session.set_mode("pure_neural")
                print(">> Switched to [PURE NEURAL] mode (cognitive routers bypassed).")
            elif len(parts) > 1 and parts[1].lower() in ["hybrid", "cognitive"]:
                session.set_mode("hybrid")
                print(">> Switched to [HYBRID COGNITIVE] mode (memory & deterministic math active).")
            else:
                print("Usage: /mode neural OR /mode hybrid")
            continue

        if user_input == "/memory":
            mem = session.get_memory_state()
            print("\n--- WORKING MEMORY STATE ---")
            print(f"Mode: {mem['mode']}")
            print(f"Total Turns: {mem['turns_count']}")
            print("Extracted Facts:")
            if mem["stored_facts"]:
                for k, v in mem["stored_facts"].items():
                    print(f"  - {k}: {v}")
            else:
                print("  (No persistent facts recorded yet)")
            print("----------------------------\n")
            continue

        if user_input == "/clear":
            session.clear_memory()
            print(">> Working memory cleared and conversation history wiped.")
            continue

        if user_input.startswith("/export"):
            parts = user_input.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else f"session_{session.session_id}.json"
            session.export_session(target)
            print(f">> Session transcript exported to {target}")
            continue

        # Execute turn
        print(f"Assistant [{session.mode.upper()}]> ", end="", flush=True)
        resp = session.chat(
            user_input,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
        )
        print(resp["response"])
        origin_tag = resp["origin"]
        lat = resp.get("latency_sec", 0.0)
        tps = resp.get("tokens_per_sec", 0.0)
        if tps > 0:
            print(f"  [Origin: {origin_tag} | Latency: {lat:.3f}s | Speed: {tps:.1f} tok/s]\n")
        else:
            print(f"  [Origin: {origin_tag} | Latency: {lat:.4f}s]\n")


if __name__ == "__main__":
    run_cli()
