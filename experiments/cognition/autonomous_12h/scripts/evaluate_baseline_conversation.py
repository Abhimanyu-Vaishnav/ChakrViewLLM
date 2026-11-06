"""
Stage 2: Deterministic Baseline Conversational Evaluation Suite
Evaluates both Parent (cbr_sft_04_step_200.pt) and Candidate (cr_exp01_stage_a_candidate_step_100.pt)
across 7 structured categories:
A. Basic Conversation
B. Context Following & Entity Updates
C. Multi-turn Coherence (5 turns)
D. Instruction Following (formats: short, list, explanation)
E. Knowledge & Uncertainty Handling
F. Multilingual Behaviour (English, Hindi, Hinglish)
G. Generation Stability & Repetition
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CAND_PATH = REPO_ROOT / "experiments" / "cognition" / "cr_exp_01" / "runs" / "stage_a_pilot" / "checkpoints" / "stage_a_candidate_step_100.pt"

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
            # Check stop condition
            should_stop = False
            for st in stop_tokens:
                if st in gen_text:
                    should_stop = True
                    break
            if should_stop:
                break
                
    full_output = tok.decode(generated_ids, errors="replace")
    return full_output.strip(), generated_ids

def run_eval(model, tok, label):
    print(f"\n--- Running Conversational Evaluation for: {label} ---")
    results = []
    for item in EVAL_SUITE:
        t0 = time.time()
        output_text, token_ids = generate_greedy(model, tok, item["prompt"], max_new_tokens=35)
        elapsed = round(time.time() - t0, 3)
        res_entry = {
            "category": item["category"],
            "id": item["id"],
            "prompt": item["prompt"],
            "generation": output_text,
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "elapsed_sec": elapsed
        }
        results.append(res_entry)
        print(f"[{item['id']}] {item['prompt'].splitlines()[-1]} -> {output_text!r} ({len(token_ids)} tokens, {elapsed}s)")
    return results

def main():
    print("=== CR-EXP-03 / AUTONOMOUS 12H — STAGE 2 CONVERSATIONAL AUDIT ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CAND_PATH)
    
    parent_results = run_eval(parent, tok, "PARENT (CBR-SFT-04 Step 200)")
    cand_results = run_eval(cand, tok, "CANDIDATE (CR-EXP-01 Stage A Step 100)")
    
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parent_checkpoint": str(PARENT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "candidate_checkpoint": str(CAND_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "parent_results": parent_results,
        "candidate_results": cand_results
    }
    
    out_json = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "baseline_conversation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved baseline evaluation results to {out_json}")

if __name__ == "__main__":
    main()
