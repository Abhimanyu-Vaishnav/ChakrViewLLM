"""
Phase 5: Context Length and Fact Position Experiments
Evaluates retrieval accuracy and target token rank as a function of:
- Fact position: Beginning, Middle, End
- Context length: Short (30 tokens), Medium (120 tokens), Long (300 tokens)
- Distractor count: 0, 2, 5 distractors
Target fact remains identical: 'Rohan lives in Nagpur.'
Target question: 'Where does Rohan live?' Expected target: 'Nagpur'.
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CANDIDATE_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

TARGET_FACT = "Rohan lives in Nagpur."
TARGET_QUERY = "\nUser: Where does Rohan live?\nAssistant:"
TARGET_ENTITY = "Nagpur"

DISTRACTORS = [
    "Priya works in Bengaluru.",
    "Amit manages the database in Mumbai.",
    "Kavita is an architect in Ahmedabad.",
    "Sunil leads the infrastructure team in Kolkata.",
    "Deepa studies electrical engineering in Chennai.",
    "Sanjay maintains the server fleet in Pune.",
    "Neha coordinates operations in Hyderabad.",
    "Vikram writes documentation in Jaipur."
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def generate_greedy(model, tok, prompt, max_new_tokens=25):
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

def analyze_target_rank(model, tok, prompt, target_str):
    tokens = tok.encode(prompt)
    target_tokens = tok.encode(target_str)
    first_target_id = target_tokens[0] if target_tokens else None
    with torch.no_grad():
        x = torch.tensor([tokens], dtype=torch.long)
        logits = model(x)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        rank = (torch.argsort(next_logits, descending=True) == first_target_id).nonzero(as_tuple=True)[0].item() + 1
        prob = probs[first_target_id].item()
    return rank, round(prob, 6)

def main():
    print("=== CR-EXP-05 Phase 5: Context Length & Position Diagnostics ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    parent = load_model(PARENT_PATH)
    cand = load_model(CANDIDATE_PATH)

    configurations = [
        # 1. Position variations with 4 distractors
        {
            "id": "POS_BEGINNING",
            "position": "beginning",
            "context": f"{TARGET_FACT} " + " ".join(DISTRACTORS[:4]),
        },
        {
            "id": "POS_MIDDLE",
            "position": "middle",
            "context": " ".join(DISTRACTORS[:2]) + f" {TARGET_FACT} " + " ".join(DISTRACTORS[2:4]),
        },
        {
            "id": "POS_END",
            "position": "end",
            "context": " ".join(DISTRACTORS[:4]) + f" {TARGET_FACT}",
        },
        # 2. Length variations (0 distractors, 2 distractors, 7 distractors)
        {
            "id": "LEN_SHORT_0_DISTRACTORS",
            "position": "only_fact",
            "context": TARGET_FACT,
        },
        {
            "id": "LEN_MEDIUM_2_DISTRACTORS",
            "position": "middle",
            "context": f"{DISTRACTORS[0]} {TARGET_FACT} {DISTRACTORS[1]}",
        },
        {
            "id": "LEN_LONG_7_DISTRACTORS",
            "position": "middle",
            "context": " ".join(DISTRACTORS[:3]) + f" {TARGET_FACT} " + " ".join(DISTRACTORS[3:7]),
        },
    ]

    results = []
    for cfg_test in configurations:
        cid = cfg_test["id"]
        ctx = cfg_test["context"]
        prompt = f"Context: {ctx}{TARGET_QUERY}"
        
        token_count = len(tok.encode(prompt))
        
        p_rank, p_prob = analyze_target_rank(parent, tok, prompt, TARGET_ENTITY)
        c_rank, c_prob = analyze_target_rank(cand, tok, prompt, TARGET_ENTITY)
        
        p_gen = generate_greedy(parent, tok, prompt)
        c_gen = generate_greedy(cand, tok, prompt)
        
        cand_found = TARGET_ENTITY.lower() in c_gen.lower()
        parent_found = TARGET_ENTITY.lower() in p_gen.lower()
        
        results.append({
            "test_id": cid,
            "position": cfg_test["position"],
            "prompt_token_count": token_count,
            "prompt": prompt,
            "target_entity": TARGET_ENTITY,
            "candidate": {
                "generated": c_gen,
                "found_target": cand_found,
                "target_rank": c_rank,
                "target_prob": c_prob
            },
            "parent": {
                "generated": p_gen,
                "found_target": parent_found,
                "target_rank": p_rank,
                "target_prob": p_prob
            }
        })
        print(f"[{cid}] Tokens: {token_count} | Position: {cfg_test['position']}")
        print(f"  Parent   : Rank={p_rank}, Prob={p_prob:.5f} | Gen: {repr(p_gen)}")
        print(f"  Candidate: Rank={c_rank}, Prob={c_prob:.5f} | Gen: {repr(c_gen)}")

    out_p = REPO_ROOT / "experiments/cognition/cr_exp_05_ood_diagnosis/context_position_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved position and context length results to {out_p}")

if __name__ == "__main__":
    main()

