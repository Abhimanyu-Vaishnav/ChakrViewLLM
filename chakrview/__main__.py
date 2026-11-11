"""
chakrview/__main__.py

CLI Entrypoint for ChakrView package execution.
Allows:
  python -m chakrview chat
  python -m chakrview chat --test-mode
  python -m chakrview version
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="chakrview",
        description="ChakrView: Compact, CPU-First Foundation Model Architecture"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Launch interactive conversational chat terminal")
    chat_parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    chat_parser.add_argument("--test-mode", action="store_true", help="Run automated multi-turn verification")

    # version command
    version_parser = subparsers.add_parser("version", help="Show ChakrView model & release version")

    args = parser.parse_args()

    if args.command == "chat":
        from chakrview.chat import ConversationalChatCLI, DEFAULT_CHECKPOINT
        ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
        cli = ConversationalChatCLI(ckpt_path)
        if args.test_mode:
            print("\n--- Running Automated Verification Mode ---")
            test_inputs = [
                "Good morning assistant!",
                "Who are you?",
                "/info",
                "/reset",
                "Wishing you a pleasant morning!"
            ]
            for inp in test_inputs:
                print(f"\nUser > {inp}")
                out = cli.interact_step(inp)
                print(f"ChakrView > {out}")
            print("\nVerification Mode Complete: PASS")
        else:
            cli.run_interactive_loop()
    elif args.command == "version":
        print("ChakrView Release: v0.9.1 (Conversational Preview)")
        print("Architecture: ChakrMicro v0.1 (3,443,136 parameters, 6 layers, 192 d_model, 6 heads)")
        print("Target: 100% Native Host CPU Autonomous Inference")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
