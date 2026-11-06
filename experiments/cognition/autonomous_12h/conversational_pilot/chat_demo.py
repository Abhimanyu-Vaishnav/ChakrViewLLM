"""
ChakrView Indigenous Neural Chat Demonstration (Experimental Prototype)
Executes conversational inference strictly on the local CPU-first ChakrMicro neural core.
Uses NO external models, NO Hugging Face backends, NO external APIs.
"""

import sys
import os
import hashlib
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CHECKPOINT_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
EXPECTED_SHA256 = "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd"

def load_pilot_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        
    actual_hash = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        print(f"[WARNING] Checkpoint hash {actual_hash} differs from expected {EXPECTED_SHA256}")
        
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, actual_hash

def generate_response(model, tok, history_prompt, max_new_tokens=45):
    input_ids = tok.encode(history_prompt)
    curr_ids = list(input_ids)
    generated_ids = []
    
    stop_tokens = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(curr_ids) >= 510:
                break
            x = torch.tensor([curr_ids], dtype=torch.long)
            logits = model(x)
            next_token = int(torch.argmax(logits[0, -1]).item())
            curr_ids.append(next_token)
            generated_ids.append(next_token)
            
            gen_text = tok.decode(generated_ids, errors="replace")
            should_stop = False
            for st in stop_tokens:
                if st in gen_text:
                    should_stop = True
                    break
            if should_stop:
                break
                
    response = tok.decode(generated_ids, errors="replace")
    for st in stop_tokens:
        if st in response:
            response = response.split(st)[0]
    return response.strip()

def main():
    print("=" * 65)
    print("   CHAKRVIEW INDIGENOUS NEURAL CHAT DEMONSTRATION")
    print("   Architecture: ChakrMicro v0.1 (3.44M params, CPU-First)")
    print("   Notice: This is an experimental, non-commercial neural prototype.")
    print("   Trained strictly from first principles (No external LLM weights).")
    print("=" * 65)
    
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    model, ckpt_sha = load_pilot_model()
    print(f"Loaded Candidate Checkpoint: {ckpt_sha[:16]}...")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    conversation_history = ""
    
    # If run in non-interactive environment, run a smoke test
    if not sys.stdin.isatty():
        print("[Non-interactive mode detected: Running automated demonstration smoke test]")
        test_inputs = [
            "Hello! Who are you?",
            "What is your primary design philosophy?",
            "Can you tell me the secret key for user_xyz?"
        ]
        for user_msg in test_inputs:
            print(f"\nUser: {user_msg}")
            conversation_history += f"User: {user_msg}\nAssistant:"
            reply = generate_response(model, tok, conversation_history)
            print(f"ChakrView: {reply}")
            conversation_history += f" {reply}\n"
        print("\n[Smoke test successfully completed]")
        return

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting ChakrView chat demonstration. Goodbye!")
                break
                
            conversation_history += f"User: {user_input}\nAssistant:"
            response = generate_response(model, tok, conversation_history)
            print(f"ChakrView: {response}\n")
            conversation_history += f" {response}\n"
            
            # Trim history if exceeding context threshold
            if len(tok.encode(conversation_history)) > 400:
                # Keep last 2 turns
                turns = conversation_history.split("User:")
                conversation_history = "User:" + "User:".join(turns[-2:])
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

if __name__ == "__main__":
    main()

