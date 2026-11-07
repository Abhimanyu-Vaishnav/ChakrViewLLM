"""
CR-EXP-11 Phase 4: Entity Binding & Working Memory Diagnostic Study
Builds an 8-level diagnostic hierarchy to assess:
- Level 1: Single fact retrieval
- Level 2: Multiple entities
- Level 3: Distractor resistance
- Level 4: Counterfactual reassignment
- Level 5: Multiple state updates
- Level 6: Interleaved entities
- Level 7: Longer conversation
- Level 8: Natural conversational variations
"""

import sys
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"

def build_8level_suite():
    levels = {}
    
    # L1: Single fact retrieval
    levels["L1_single_fact"] = [
        {"prompt": "Context: Meera lives in Jaipur.\nQuestion: Where does Meera live?\nAnswer: ", "target": "Jaipur", "subject": "Meera", "distractors": []},
        {"prompt": "Context: Aarav works in Pune.\nQuestion: Where does Aarav work?\nAnswer: ", "target": "Pune", "subject": "Aarav", "distractors": []},
        {"prompt": "Context: Kabir lives in Bhopal.\nQuestion: Where does Kabir live?\nAnswer: ", "target": "Bhopal", "subject": "Kabir", "distractors": []},
        {"prompt": "Context: Manish works in Indore.\nQuestion: Where does Manish work?\nAnswer: ", "target": "Indore", "subject": "Manish", "distractors": []},
        {"prompt": "Context: Kavita lives in Amritsar.\nQuestion: Where does Kavita live?\nAnswer: ", "target": "Amritsar", "subject": "Kavita", "distractors": []},
    ]
    
    # L2: Multiple entities (2 entities, query both)
    levels["L2_multiple_entities"] = [
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune.\nQuestion: Where does Meera work?\nAnswer: ", "target": "Jaipur", "subject": "Meera", "distractors": ["Pune"]},
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune.\nQuestion: Where does Aarav work?\nAnswer: ", "target": "Pune", "subject": "Aarav", "distractors": ["Jaipur"]},
        {"prompt": "Context: Kabir lives in Bhopal. Manish lives in Indore.\nQuestion: Where does Kabir live?\nAnswer: ", "target": "Bhopal", "subject": "Kabir", "distractors": ["Indore"]},
        {"prompt": "Context: Kabir lives in Bhopal. Manish lives in Indore.\nQuestion: Where does Manish live?\nAnswer: ", "target": "Indore", "subject": "Manish", "distractors": ["Bhopal"]},
    ]
    
    # L3: Distractor resistance (3 entities)
    levels["L3_distractor_resistance"] = [
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune. Rohan works in Delhi.\nQuestion: Where does Meera work?\nAnswer: ", "target": "Jaipur", "subject": "Meera", "distractors": ["Pune", "Delhi"]},
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune. Rohan works in Delhi.\nQuestion: Where does Rohan work?\nAnswer: ", "target": "Delhi", "subject": "Rohan", "distractors": ["Jaipur", "Pune"]},
        {"prompt": "Context: Kavita is in Patna. Divya is in Surat. Tanvi is in Kochi.\nQuestion: Where is Divya located?\nAnswer: ", "target": "Surat", "subject": "Divya", "distractors": ["Patna", "Kochi"]},
    ]
    
    # L4: Counterfactual reassignment (single update)
    levels["L4_counterfactual_update"] = [
        {"prompt": "Context: Meera works in Jaipur. Meera later moves to Bhopal.\nQuestion: Where does Meera work now?\nAnswer: ", "target": "Bhopal", "subject": "Meera", "distractors": ["Jaipur"]},
        {"prompt": "Context: Kabir lives in Pune. Later, Kabir relocated to Indore.\nQuestion: Where does Kabir live now?\nAnswer: ", "target": "Indore", "subject": "Kabir", "distractors": ["Pune"]},
        {"prompt": "Context: Manish started in Delhi. Manish has moved to Amritsar.\nQuestion: Where is Manish now?\nAnswer: ", "target": "Amritsar", "subject": "Manish", "distractors": ["Delhi"]},
    ]
    
    # L5: Multiple state updates (A -> B -> C)
    levels["L5_multiple_updates"] = [
        {"prompt": "Context: Meera works in Jaipur. She moves to Bhopal. Later she moves to Indore.\nQuestion: Where does Meera work now?\nAnswer: ", "target": "Indore", "subject": "Meera", "distractors": ["Jaipur", "Bhopal"]},
        {"prompt": "Context: Aarav was residing in Pune. He shifted to Lucknow. Finally he relocated to Goa.\nQuestion: Where does Aarav live now?\nAnswer: ", "target": "Goa", "subject": "Aarav", "distractors": ["Pune", "Lucknow"]},
    ]
    
    # L6: Interleaved entities
    levels["L6_interleaved_entities"] = [
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune. Meera moves to Bhopal. Aarav moves to Delhi.\nQuestion: Where does Meera work now?\nAnswer: ", "target": "Bhopal", "subject": "Meera", "distractors": ["Jaipur", "Pune", "Delhi"]},
        {"prompt": "Context: Meera works in Jaipur. Aarav works in Pune. Meera moves to Bhopal. Aarav moves to Delhi.\nQuestion: Where does Aarav work now?\nAnswer: ", "target": "Delhi", "subject": "Aarav", "distractors": ["Jaipur", "Pune", "Bhopal"]},
    ]
    
    # L7: Longer conversation
    levels["L7_longer_conversation"] = [
        {"prompt": "User: Hello! Please keep track of our team.\nAssistant: Understood, I will track them.\nUser: Meera has joined the Jaipur branch.\nAssistant: Noted that Meera is in Jaipur.\nUser: Aarav joined Pune.\nAssistant: Noted that Aarav is in Pune.\nUser: Meera just transferred to Chandigarh.\nAssistant: Updated Meera's location to Chandigarh.\nUser: Where is Meera working currently?\nAssistant: ", "target": "Chandigarh", "subject": "Meera", "distractors": ["Jaipur", "Pune"]},
        {"prompt": "User: Hello! Please keep track of our team.\nAssistant: Understood, I will track them.\nUser: Meera has joined the Jaipur branch.\nAssistant: Noted that Meera is in Jaipur.\nUser: Aarav joined Pune.\nAssistant: Noted that Aarav is in Pune.\nUser: Meera just transferred to Chandigarh.\nAssistant: Updated Meera's location to Chandigarh.\nUser: Where is Aarav working currently?\nAssistant: ", "target": "Pune", "subject": "Aarav", "distractors": ["Jaipur", "Chandigarh"]},
    ]
    
    # L8: Natural conversational variations
    levels["L8_natural_variations"] = [
        {"prompt": "User: Hey ChakrView, can you tell me where Kavita is staying? She originally arrived in Lucknow but unpacked her bags in Bhopal today.\nAssistant: ", "target": "Bhopal", "subject": "Kavita", "distractors": ["Lucknow"]},
        {"prompt": "User: I need to contact Manish. He left his old office in Patna and is stationed at our Kochi branch right now.\nAssistant: ", "target": "Kochi", "subject": "Manish", "distractors": ["Patna"]},
    ]
    
    return levels

def evaluate_levels(model, tok, levels_dict):
    out = {}
    abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state"]
    
    for lvl_name, probes in levels_dict.items():
        hits = 0
        distractor_errors = 0
        abstentions = 0
        details = []
        
        for p in probes:
            prompt = p["prompt"]
            tgt = p["target"]
            distractors = p["distractors"]
            
            enc = tok.encode(prompt)
            curr = list(enc)
            with torch.no_grad():
                for _ in range(25):
                    x = torch.tensor([curr], dtype=torch.long)
                    logits = model(x)[0, -1]
                    nxt = logits.argmax().item()
                    curr.append(nxt)
                    if nxt == 13 and len(curr) - len(enc) > 3:
                        break
            gen = tok.decode(curr[len(enc):], errors="replace").strip()
            
            is_abstained = any(c in gen.lower() for c in abstain_cues)
            has_tgt = tgt.lower() in gen.lower()
            has_dist = any(d.lower() in gen.lower() for d in distractors)
            
            if is_abstained:
                abstentions += 1
            elif has_dist and not has_tgt:
                distractor_errors += 1
            elif has_tgt and not has_dist:
                hits += 1
            elif has_tgt and has_dist:
                # Contaminated / ambiguous
                hits += 1
                
            details.append({
                "prompt": prompt,
                "target": tgt,
                "generated": gen,
                "is_hit": (has_tgt and not (has_dist and not has_tgt)),
                "is_abstained": is_abstained,
                "has_distractor": has_dist
            })
            
        out[lvl_name] = {
            "total": len(probes),
            "hits": hits,
            "accuracy_pct": round(hits / len(probes) * 100, 2),
            "distractor_errors": distractor_errors,
            "abstentions": abstentions,
            "details": details
        }
    return out

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    levels = build_8level_suite()
    
    models = {
        "cr09_candidate_3": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt",
        "cr10_arm_b": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_arm_b/cr_exp_10_arm_b_best_val_copy.pt"
    }
    
    results = {}
    for name, ckpt in models.items():
        print(f"\n--- Evaluating 8-Level Hierarchy on {name} ---")
        cfg = ModelConfig()
        m = ChakrMicro(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("model_state_dict", ck))
        m.eval()
        
        res = evaluate_levels(m, tok, levels)
        results[name] = res
        for lvl, data in res.items():
            print(f"  {lvl:28s}: {data['accuracy_pct']:5.2f}% ({data['hits']}/{data['total']}) | Distractor Errs: {data['distractor_errors']} | Abstentions: {data['abstentions']}")
            
    out_file = OUT_DIR / "entity_binding_diagnostic_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 8-level diagnostic results to {out_file}")

if __name__ == "__main__":
    main()

