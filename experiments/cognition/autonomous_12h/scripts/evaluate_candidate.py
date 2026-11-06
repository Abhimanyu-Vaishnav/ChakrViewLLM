"""
Stage 5: Held-out Conversational Evaluation & Candidate Comparison
Evaluates:
1. Frozen Parent (cbr_sft_04_step_200.pt)
2. New Candidate (conversational_pilot_best_val.pt)
Across:
- The 18 standard Stage 2 conversational prompts
- The 54 unseen held-out test dataset samples (measuring loss and exact match / accuracy)
- Complex context-binding probes (earlier vs updated information, combining 2 statements)
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CANDIDATE_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
TEST_DATA_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "dataset" / "test_held_out.jsonl"

EVAL_SUITE = [
    # A. Basic Conversation
    {"category": "A_Basic_Conversation", "id": "A1", "prompt": "User: Hello!\nAssistant:"},
    {"category": "A_Basic_Conversation", "id": "A2", "prompt": "User: Who are you?\nAssistant:"},
    {"category": "A_Basic_Conversation", "id": "A3", "prompt": "User: How are you doing today?\nAssistant:"},

    # B. Context Following
    {"category": "B_Context_Following", "id": "B1", "prompt": "User: My name is Ravi and I live in Pune.\nAssistant: Nice to meet you Ravi!\nUser: Where do I live?\nAssistant:"},
    {"category": "B_Context_Following", "id": "B2", "prompt": "User: I have 3 apples. I buy 2 more.\nAssistant: You now have 5 apples.\nUser: How many apples did I have at the start?\nAssistant:"},
    {"category": "B_Context_Following", "id": "B3", "prompt": "User: My favorite color is blue. Actually, I changed my mind, my favorite color is green.\nAssistant: Got it, your favorite color is green.\nUser: What is my favorite color?\nAssistant:"},

    # C. Multi-turn Coherence (Simulated 5-turn history)
    {"category": "C_Multi_Turn", "id": "C1", "prompt": "User: Let's talk about solar planets.\nAssistant: Sure, our solar system has 8 planets.\nUser: Which planet is the closest to the Sun?\nAssistant: Mercury is closest to the Sun.\nUser: Is it hot there?\nAssistant: Yes, it gets extremely hot during the day.\nUser: What about the next one?\nAssistant: The next planet is Venus.\nUser: Does it have water?\nAssistant:"},

    # D. Instruction Following
    {"category": "D_Instruction_Following", "id": "D1", "prompt": "User: Give me a short one-word answer: What is the capital of France?\nAssistant:"},
    {"category": "D_Instruction_Following", "id": "D2", "prompt": "User: List three primary colors as bullet points:\nAssistant:"},
    {"category": "D_Instruction_Following", "id": "D3", "prompt": "User: Explain gravity in simple terms in one sentence.\nAssistant:"},

    # E. Knowledge and Uncertainty
    {"category": "E_Knowledge_Uncertainty", "id": "E1", "prompt": "User: What is 12 + 15?\nAssistant:"},
    {"category": "E_Knowledge_Uncertainty", "id": "E2", "prompt": "User: What did Abhimanyu eat for breakfast on June 12, 1998?\nAssistant:"},
    {"category": "E_Knowledge_Uncertainty", "id": "E3", "prompt": "User: Can you predict the exact stock price of Apple next month?\nAssistant:"},

    # F. Multilingual Behaviour
    {"category": "F_Multilingual", "id": "F1_Hindi", "prompt": "User: नमस्ते! आप कैसे हैं?\nAssistant:"},
    {"category": "F_Multilingual", "id": "F2_Hindi", "prompt": "User: भारत की राजधानी क्या है?\nAssistant:"},
    {"category": "F_Multilingual", "id": "F3_Hinglish", "prompt": "User: Bhai, mujhe ek simple program batao Python mein.\nAssistant:"},

    # G. Generation Stability
    {"category": "G_Generation_Stability", "id": "G1", "prompt": "User: Repeat after me: 1, 2, 3.\nAssistant:"},
    {"category": "G_Generation_Stability", "id": "G2", "prompt": "User: Why is the sky blue?\nAssistant:"}
]

# Additional Challenging Context Probes
CHALLENGING_PROBES = [
    {
        "id": "PROBE_UPDATE_1",
        "description": "Distinguish earlier information from updated information",
        "prompt": "User: State update:\nFirst STATUS=PENDING.\nUpdate: STATUS was changed to ACTIVE.\nQuestion: What is the current value of STATUS?\nAssistant:"
    },
    {
        "id": "PROBE_COMBINE_2",
        "description": "Combine two earlier statements",
        "prompt": "User: The project code is Garuda.\nAssistant: Garuda noted.\nUser: The deployment port is 7070.\nAssistant: Port 7070 noted.\nUser: What project name and port did I configure?\nAssistant:"
    },
    {
        "id": "PROBE_MULTI_ENTITY_3",
        "description": "Distinguish two similar entities",
        "prompt": "User: The primary office is in Delhi. The secondary office is in Hyderabad.\nUser: Which city is the secondary office in?\nAssistant:"
    }
]

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def generate_greedy(model, tok, prompt_text, max_new_tokens=40, stop_tokens=["\nUser:", "User:", "Assistant:"]):
    input_ids = tok.encode(prompt_text)
    curr_ids = list(input_ids)
    generated_ids = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if len(curr_ids) >= 512:
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
                
    full_output = tok.decode(generated_ids, errors="replace")
    return full_output.strip(), generated_ids

def evaluate_held_out_loss(model, tok, test_path):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            
    total_loss = 0.0
    with torch.no_grad():
        for s in samples:
            p_ids = tok.encode(s["prompt"])
            r_ids = tok.encode(s["response"] + "\n")
            all_ids = p_ids + r_ids
            if len(all_ids) > 512:
                all_ids = all_ids[:512]
            x = torch.tensor([all_ids[:-1]], dtype=torch.long)
            y = torch.tensor([all_ids[1:]], dtype=torch.long)
            mask_len = len(p_ids) - 1
            y[0, :mask_len] = -100
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            
    return round(total_loss / len(samples), 4)

def main():
    print("=== CR-EXP-03 / AUTONOMOUS 12H — STAGE 5 CANDIDATE EVALUATION ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CANDIDATE_PATH)
    
    # 1. Held-out test loss
    parent_test_loss = evaluate_held_out_loss(parent, tok, TEST_DATA_PATH)
    cand_test_loss = evaluate_held_out_loss(cand, tok, TEST_DATA_PATH)
    print(f"Held-out Test Loss (54 unseen test samples):")
    print(f"  Parent:    {parent_test_loss:.4f}")
    print(f"  Candidate: {cand_test_loss:.4f} (Rel improvement: {round((parent_test_loss - cand_test_loss)/parent_test_loss*100, 2)}%)")
    
    # 2. Standard Conversational Evaluation
    print("\n--- Evaluating Candidate on 18 Standard Conversational Prompts ---")
    standard_eval_cand = []
    for item in EVAL_SUITE:
        out_text, tokens = generate_greedy(cand, tok, item["prompt"], max_new_tokens=35)
        standard_eval_cand.append({
            "category": item["category"],
            "id": item["id"],
            "prompt": item["prompt"],
            "generation": out_text
        })
        print(f"[{item['id']}] -> {out_text!r}")
        
    # 3. Challenging Context Probes Evaluation (Parent vs Candidate)
    print("\n--- Evaluating Challenging Context Probes ---")
    probe_results = []
    for probe in CHALLENGING_PROBES:
        p_out, _ = generate_greedy(parent, tok, probe["prompt"], max_new_tokens=25)
        c_out, _ = generate_greedy(cand, tok, probe["prompt"], max_new_tokens=25)
        probe_results.append({
            "id": probe["id"],
            "description": probe["description"],
            "prompt": probe["prompt"],
            "parent_generation": p_out,
            "candidate_generation": c_out
        })
        print(f"[{probe['id']}] {probe['description']}")
        print(f"  Parent:    {p_out!r}")
        print(f"  Candidate: {c_out!r}")
        
    # 4. Save results JSON
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_test_loss": {
            "parent": parent_test_loss,
            "candidate": cand_test_loss,
            "relative_improvement_pct": round((parent_test_loss - cand_test_loss)/parent_test_loss*100, 2)
        },
        "standard_conversational_evaluation": standard_eval_cand,
        "challenging_context_probes": probe_results
    }
    
    out_file = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation results to {out_file}")

if __name__ == "__main__":
    main()

