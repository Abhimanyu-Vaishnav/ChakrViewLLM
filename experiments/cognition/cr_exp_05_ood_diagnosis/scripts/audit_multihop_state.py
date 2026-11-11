"""
Phase 6: Multi-Hop and Dynamic-State Experiments
Tests separately:
A. Single-hop retrieval
B. Two-hop retrieval (Company -> City)
C. State update (PENDING -> ACTIVE)
D. Contradiction (ACTIVE vs INACTIVE)
E. Multiple entities (distinguishing 2 people and 2 cities)
F. Long-range follow-up (intervening conversational turns)
"""

import json
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CANDIDATE_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

MULTIHOP_CASES = [
    {
        "id": "CASE_A_SINGLE_HOP",
        "type": "single_hop",
        "description": "Direct single-hop context retrieval",
        "prompt": "Context: Ravi lives in Pune.\nUser: Where does Ravi live?\nAssistant:",
        "target": "Pune"
    },
    {
        "id": "CASE_B_TWO_HOP",
        "type": "two_hop",
        "description": "Two-hop compositional inference (Ravi -> Garuda Systems -> Jaipur)",
        "prompt": "Context: Ravi owns Garuda Systems. Garuda Systems is located in Jaipur.\nUser: In which city is Ravi's company located?\nAssistant:",
        "target": "Jaipur"
    },
    {
        "id": "CASE_C_STATE_UPDATE",
        "type": "state_update",
        "description": "Overwriting previous value with new state",
        "prompt": "Context: Order status was PENDING. Later, the status was updated to ACTIVE.\nUser: What is the current order status?\nAssistant:",
        "target": "ACTIVE"
    },
    {
        "id": "CASE_D_CONTRADICTION",
        "type": "contradiction",
        "description": "Explicit conflict between earlier and later assertions",
        "prompt": "Context: Report Alpha claims system is SECURE. Report Beta claims system is VULNERABLE.\nUser: What does Report Beta say about the system?\nAssistant:",
        "target": "VULNERABLE"
    },
    {
        "id": "CASE_E_MULTI_ENTITY",
        "type": "multi_entity",
        "description": "Entity disambiguation (Pooja vs Ankit, Mumbai vs Delhi)",
        "prompt": "Context: Pooja works in Mumbai. Ankit works in Delhi.\nUser: Which city does Ankit work in?\nAssistant:",
        "target": "Delhi"
    },
    {
        "id": "CASE_F_LONG_RANGE_FOLLOWUP",
        "type": "long_range_followup",
        "description": "Querying fact introduced 3 turns earlier",
        "prompt": (
            "User: My project codename is Phoenix.\n"
            "Assistant: Phoenix recorded.\n"
            "User: Let's discuss database indexes.\n"
            "Assistant: Indexes speed up query performance.\n"
            "User: We will use B-tree indexes.\n"
            "Assistant: B-tree is a solid choice.\n"
            "User: What was my project codename?\n"
            "Assistant:"
        ),
        "target": "Phoenix"
    }
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def generate_greedy(model, tok, prompt, max_new_tokens=30):
    tokens = tok.encode(prompt)
    generated = []
    stop_tokens = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([tokens], dtype=torch.long)
            logits = model(inp)
            next_token = torch.argmax(logits[0, -1, :]).item()
            tokens.append(next_token)
            generated.append(next_token)
            text = tok.decode(generated, errors="replace")
            if any(st in text for st in stop_tokens):
                break
    return tok.decode(generated, errors="replace").strip()

def main():
    print("=== CR-EXP-05 Phase 6: Multi-Hop and Dynamic-State Experiments ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    parent = load_model(PARENT_PATH)
    cand = load_model(CANDIDATE_PATH)

    results = []
    for c in MULTIHOP_CASES:
        cid = c["id"]
        prompt = c["prompt"]
        target = c["target"]
        
        p_gen = generate_greedy(parent, tok, prompt)
        c_gen = generate_greedy(cand, tok, prompt)
        
        c_success = target.lower() in c_gen.lower()
        p_success = target.lower() in p_gen.lower()
        
        results.append({
            "case_id": cid,
            "type": c["type"],
            "description": c["description"],
            "prompt": prompt,
            "target": target,
            "candidate_generation": c_gen,
            "candidate_success": c_success,
            "parent_generation": p_gen,
            "parent_success": p_success
        })
        print(f"[{cid}] Target: {target}")
        print(f"  Parent   : Success={p_success} | Gen: {repr(p_gen)}")
        print(f"  Candidate: Success={c_success} | Gen: {repr(c_gen)}")

    out_p = REPO_ROOT / "experiments/cognition/cr_exp_05_ood_diagnosis/multihop_state_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved multi-hop & state results to {out_p}")

if __name__ == "__main__":
    main()

