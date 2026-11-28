"""
ChakrView Interactive Chat CLI (Pure Neural Preview)
Run:
    python chat_cli.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
from chakrview.inference_package import ChakrViewGenerator, ChakrChatSession

def main():
    ckpt_path = "experiments/cv24h_release/chakr_38m_conversational_sft.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt"

    print("=============================================================")
    print(" CHAKRVIEW CONVERSATIONAL RESEARCH PREVIEW (ALPHA)")
    print(f" Loaded Checkpoint: {ckpt_path}")
    print(" Note: This is a pure neural experimental research preview.")
    print(" Type 'quit', 'exit', or 'reset' to manage your session.")
    print("=============================================================\n")

    generator = ChakrViewGenerator(ckpt_path)
    session = ChakrChatSession(generator)

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting ChakrView chat session. Goodbye!")
                break
            if user_input.lower() == "reset":
                session.reset()
                print("[Session history reset]\n")
                continue

            response = session.chat(user_input, max_new_tokens=64, temperature=0.7, top_k=40, repetition_penalty=1.15)
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error during generation: {e}\n")

if __name__ == "__main__":
    main()
