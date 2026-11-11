"""
Phase 4: Controlled Variable-Binding Diagnostic Experiment
Tests controlled synthetic key-value facts:
- Forward query (What does Quasar map to?)
- Reverse query (Which entity maps to Rhodium?)
- Negation / counterfactual (Does Quasar map to Platinum?)
- Dynamic update (Quasar updated from Rhodium to Osmium)
- Similar name interference (Quasar vs Quasor)
- Irrelevant distractor injection
Measures: exact output, target log prob, top-1 & top-5 tokens, entropy, candidate vs parent.
"""

import json
import math
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

VARIABLE_BINDING_PROBES = [
    # 1. Forward synthetic binding (unseen entities)
    {
        "id": "VB_01_FORWARD",
        "description": "Forward synthetic key-value retrieval",
        "prompt": "Context: Quasar is mapped to Rhodium.\nUser: What does Quasar map to?\nAssistant:",
        "target_entity": "Rhodium",
    },
    # 2. Reverse synthetic binding (unseen entities)
    {
        "id": "VB_02_REVERSE",
        "description": "Reverse synthetic key-value retrieval",
        "prompt": "Context: Quasar is mapped to Rhodium.\nUser: Which entity is mapped to Rhodium?\nAssistant:",
        "target_entity": "Quasar",
    },
    # 3. Negation query
    {
        "id": "VB_03_NEGATION",
        "description": "Querying unmapped alternative entity",
        "prompt": "Context: Quasar is mapped to Rhodium.\nUser: Does Quasar map to Platinum?\nAssistant:",
        "target_entity": "No",
    },
    # 4. Dynamic value update
    {
        "id": "VB_04_DYNAMIC_UPDATE",
        "description": "Sequential value reassignment",
        "prompt": "Context: First, Quasar is mapped to Rhodium. Later, Quasar is updated to Osmium.\nUser: What is the current mapping for Quasar?\nAssistant:",
        "target_entity": "Osmium",
    },
    # 5. Similar name interference
    {
        "id": "VB_05_NAME_INTERFERENCE",
        "description": "Interference between phonetically/orthographically close keys",
        "prompt": "Context: Quasar is mapped to Rhodium. Quasor is mapped to Platinum.\nUser: What does Quasar map to?\nAssistant:",
        "target_entity": "Rhodium",
    },
    # 6. Irrelevant distractor
    {
        "id": "VB_06_DISTRACTOR",
        "description": "Synthetic binding in presence of unrelated distractor facts",
        "prompt": "Context: Solar flare was recorded in Sector 4. Quasar is mapped to Rhodium. Atmospheric pressure is 101 kPa.\nUser: What does Quasar map to?\nAssistant:",
        "target_entity": "Rhodium",
    },
    # 7. Familiar entities, synthetic template
    {
        "id": "VB_07_FAMILIAR_ENTITIES",
        "description": "Familiar entities (Ravi, Pune) mapped under synthetic relational phrasing",
        "prompt": "Context: Entity Ravi is bound to value Pune.\nUser: What value is entity Ravi bound to?\nAssistant:",
        "target_entity": "Pune",
    },
    # 8. Paired counterfactual control (swap targets)
    {
        "id": "VB_08_PAIRED_SWAP",
        "description": "Identical structure with mapped target swapped to Titanium",
        "prompt": "Context: Quasar is mapped to Titanium.\nUser: What does Quasar map to?\nAssistant:",
        "target_entity": "Titanium",
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

def analyze_logits(model, tok, prompt, target_str):
    tokens = tok.encode(prompt)
    # The actual entity content tokens (without the leading space separator)
    target_tokens = tok.encode(target_str)
    first_target_id = target_tokens[0] if target_tokens else None
    
    with torch.no_grad():
        x = torch.tensor([tokens], dtype=torch.long)
        logits = model(x)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        
        # Entropy
        log_probs = F.log_softmax(next_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        
        # Target token logprob
        target_logprob = None
        target_prob = None
        target_rank = None
        if first_target_id is not None:
            target_prob = probs[first_target_id].item()
            target_logprob = log_probs[first_target_id].item()
            target_rank = (torch.argsort(next_logits, descending=True) == first_target_id).nonzero(as_tuple=True)[0].item() + 1
            
        # Top-5 tokens
        top5_indices = torch.topk(next_logits, k=5).indices.tolist()
        top5_tokens = [{"token_id": tid, "token_str": repr(tok.decode([tid], errors="replace")), "prob": round(probs[tid].item(), 4)} for tid in top5_indices]
        
    return {
        "entropy": round(entropy, 4),
        "target_first_token_id": first_target_id,
        "target_prob": round(target_prob, 6) if target_prob else 0.0,
        "target_logprob": round(target_logprob, 4) if target_logprob else -999.0,
        "target_rank": target_rank,
        "top5_predictions": top5_tokens,
    }

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
    print("=== CR-EXP-05 Phase 4: Controlled Variable-Binding Experiments ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    parent = load_model(PARENT_PATH)
    cand = load_model(CANDIDATE_PATH)

    results = []
    for probe in VARIABLE_BINDING_PROBES:
        pid = probe["id"]
        prompt = probe["prompt"]
        target = probe["target_entity"]
        
        p_stats = analyze_logits(parent, tok, prompt, target)
        c_stats = analyze_logits(cand, tok, prompt, target)
        
        p_gen = generate_greedy(parent, tok, prompt)
        c_gen = generate_greedy(cand, tok, prompt)
        
        cand_success = (target.lower() in c_gen.lower())
        parent_success = (target.lower() in p_gen.lower())
        
        results.append({
            "probe_id": pid,
            "description": probe["description"],
            "prompt": prompt,
            "target_entity": target,
            "candidate": {
                "generated_text": c_gen,
                "success": cand_success,
                "logits_analysis": c_stats
            },
            "parent": {
                "generated_text": p_gen,
                "success": parent_success,
                "logits_analysis": p_stats
            }
        })
        print(f"[{pid}] Target: {target}")
        print(f"  Parent   : {repr(p_gen)} (Top-1: {p_stats['top5_predictions'][0]['token_str']}, Target rank: {p_stats['target_rank']})")
        print(f"  Candidate: {repr(c_gen)} (Top-1: {c_stats['top5_predictions'][0]['token_str']}, Target rank: {c_stats['target_rank']})")

    out_p = REPO_ROOT / "experiments/cognition/cr_exp_05_ood_diagnosis/variable_binding_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved variable binding results to {out_p}")

if __name__ == "__main__":
    main()
