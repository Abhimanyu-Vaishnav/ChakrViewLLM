"""
chakrview/cli/interactive_chat.py
ChakrView 1.0 Release Candidate: Interactive Terminal Chat & Manual Test Interface.
Supports interactive multi-turn dialogue, tool provenance inspection, self-model queries,
memory viewing, and health check diagnostics.
"""

import sys
import os
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chakrview.app.brain_runtime import ChakrView10Brain

def start_interactive_session():
    print("=" * 80)
    print("         CHAKRVIEW 1.0 RELEASE CANDIDATE (RC1) INTERACTIVE CHAT")
    print("=" * 80)
    print("Initializing Sovereign Cognitive Brain Runtime...")
    brain = ChakrView10Brain()
    health = brain.self_healing_check()
    print(f"Health Check: {health['status']} | Hardware: {brain.device.upper()}")
    print("Type your message below. Commands: '/exit' to quit, '/health' for diagnostics, '/self' for model specs.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                print("\nEnding ChakrView 1.0 session. Sovereign local brain shutdown.")
                break

            if user_input.lower() == "/health":
                h = brain.self_healing_check()
                print(f"[System Health]: Status={h['status']}, Issues={h['issues']}")
                continue

            if user_input.lower() == "/self":
                sm = brain.get_self_model()
                print("[Operational Self-Model]:")
                for k, v in sm.items():
                    print(f"  {k}: {v}")
                continue

            # Process turn
            res = brain.generate(user_input)
            print(f"\nChakrView [{res['mode']}] > {res['text']}\n")
            if res.get("provenance") and res["mode"] != "pure_neural":
                print(f"  [Provenance]: {res['provenance']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    start_interactive_session()
