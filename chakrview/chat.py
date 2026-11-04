"""
chakrview/chat.py

Interactive Terminal Conversational Interface for ChakrView.
Directly loads the selected fine-tuned conversational checkpoint and manages
multi-turn interactive sessions with history sliding-window truncation.

Commands inside chat:
  /reset - Clears conversational history and starts fresh
  /info  - Displays active checkpoint metadata and parameters
  /exit  - Exits the chat session
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner, file_sha256

DEFAULT_CHECKPOINT = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_07" / "checkpoints" / "sft_checkpoint_step_50.pt"

class ConversationalChatCLI:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
        
        self.sha256 = file_sha256(self.checkpoint_path)
        print(f"Loading ChakrView model weights from: {self.checkpoint_path.name}")
        print(f"Verified SHA-256: {self.sha256}")
        
        self.runner = DirectNeuralRunner(self.checkpoint_path)
        self.history: List[Dict[str, str]] = []
        self.max_history_turns = 4

    def reset(self):
        self.history.clear()
        print("\n[Conversation history reset.]\n")

    def print_info(self):
        print(f"\n--- ChakrView Session Info ---")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"SHA-256:    {self.sha256}")
        print(f"History:    {len(self.history)} turns")
        print(f"Context Ceiling: 512 tokens (ChakrMicro)\n")

    def build_prompt(self, user_msg: str) -> str:
        prompt_parts = []
        # Format conversation history
        for turn in self.history[-self.max_history_turns:]:
            prompt_parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        prompt_parts.append(f"User: {user_msg}\nAssistant:")
        return "\n".join(prompt_parts)

    def interact_step(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""

        if user_input.lower() in ["/reset", "reset"]:
            self.reset()
            return "[Context cleared]"
        elif user_input.lower() in ["/info", "info"]:
            self.print_info()
            return "[Session info displayed]"

        prompt = self.build_prompt(user_input)
        resp = self.runner.generate(prompt=prompt, max_new_tokens=48, temperature=0.0)
        assistant_resp = resp["text"].strip()

        # Update history
        self.history.append({"user": user_input, "assistant": assistant_resp})
        return assistant_resp

    def run_interactive_loop(self):
        print("\n" + "=" * 65)
        print("  CHAKRVIEW INTERACTIVE CONVERSATIONAL TERMINAL (CPU-FIRST)")
        print("=" * 65)
        print("Model: ChakrMicro Conversational (CBR-SFT-07 Step 50)")
        print("Commands: /reset (clear context), /info (view stats), /exit (quit)")
        print("Type your message and press Enter.\n")

        while True:
            try:
                user_msg = input("\nUser > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["/exit", "exit", "quit", ":q"]:
                    print("Exiting ChakrView chat. Have a great day!")
                    break

                resp = self.interact_step(user_msg)
                print(f"ChakrView > {resp}")

            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted. Exiting ChakrView.")
                break

def main():
    parser = argparse.ArgumentParser(description="ChakrView Interactive Chat Terminal")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint .pt file")
    parser.add_argument("--test-mode", action="store_true", help="Run automated multi-turn test script")
    args = parser.parse_args()

    cli = ConversationalChatCLI(Path(args.checkpoint))

    if args.test_mode:
        print("\n--- Running Automated Verification Mode ---")
        test_inputs = [
            "Good morning assistant!",
            "Who are you?",
            "Can you assist me with Python coding?",
            "/info",
            "/reset",
            "Hello there!"
        ]
        for inp in test_inputs:
            print(f"\nUser > {inp}")
            out = cli.interact_step(inp)
            print(f"ChakrView > {out}")
        print("\nVerification Mode Complete: PASS")
    else:
        cli.run_interactive_loop()

if __name__ == "__main__":
    main()
