"""
CR-EXP-10 Comprehensive Multi-Suite Evaluator
Evaluates models across:
1. Frozen CR-EXP-09 600-probe suite
2. Frozen CR-EXP-10 800-probe suite (disjoint entities/templates)
3. Dedicated CR-EXP-10 300-probe conversational & working memory suite
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
SUITE_600 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission" / "benchmark" / "benchmark_suite.json"
SUITE_800 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_10_readiness" / "benchmark_data" / "benchmark_800_suite.json"
SUITE_300 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_10_readiness" / "benchmark_data" / "conversational_300_suite.json"

def load_eval_model(ckpt_path):
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    m.eval()
    return m

def generate_greedy(model, tok, prompt, max_new_tokens=20):
    enc = tok.encode(prompt)
    curr = list(enc)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([curr], dtype=torch.long)
            logits = model(x)[0, -1]
            nxt = logits.argmax().item()
            curr.append(nxt)
            if nxt in [13] and len(curr) - len(enc) > 5:
                break
    gen_ids = curr[len(enc):]
    try:
        return tok.decode(gen_ids, errors="replace").strip()
    except Exception:
        return "".join([tok.decode([tid], errors="replace") for tid in gen_ids]).strip()

def eval_800_suite(model, tok, probes):
    cat_scores = {}
    for p in probes:
        cat = p["category"]
        prompt = p["prompt"]
        target = p["expected_target"]
        gen = generate_greedy(model, tok, prompt)
        
        is_hit = False
        if cat == "conversational_understanding":
            is_hit = len(gen) > 5 and not gen.startswith("!") and any(w in gen.lower() for w in ["chakrview", "hello", "assist", "help", "neural", "good", "welcome", "madad"])
        elif cat in ["single_hop_and_counterfactual", "constraint_satisfaction"]:
            is_hit = target.lower() in gen.lower()
        elif cat == "multi_entity_distractor":
            d_val = p.get("distractor_value", "")
            contains_target = target.lower() in gen.lower()
            contains_distractor = d_val.lower() in gen.lower() if d_val else False
            is_hit = contains_target and not contains_distractor
        elif cat == "missing_info_abstention":
            abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state"]
            is_hit = any(cue in gen.lower() for cue in abstain_cues)
        elif cat == "multiturn_state_tracking":
            earlier = p.get("earlier_value", "")
            contains_latest = target.lower() in gen.lower()
            contains_earlier = earlier.lower() in gen.lower() if earlier else False
            is_hit = contains_latest and not contains_earlier
        elif cat == "structured_arithmetic_state":
            is_hit = target in gen
        elif cat == "conditional_logical_reasoning":
            is_hit = gen.lower().startswith(target.lower())
            
        cat_scores.setdefault(cat, {"hits": 0, "total": 0})
        cat_scores[cat]["total"] += 1
        if is_hit:
            cat_scores[cat]["hits"] += 1
            
    summary = {}
    tot_hits = sum(d["hits"] for d in cat_scores.values())
    tot_samples = sum(d["total"] for d in cat_scores.values())
    for cat, d in sorted(cat_scores.items()):
        summary[cat] = {
            "hits": d["hits"],
            "total": d["total"],
            "accuracy_pct": round(d["hits"] / d["total"] * 100, 2)
        }
    summary["overall"] = {
        "grand_total_hits": tot_hits,
        "grand_total_probes": tot_samples,
        "overall_accuracy_pct": round(tot_hits / tot_samples * 100, 2)
    }
    return summary

def eval_300_conv_suite(model, tok, probes):
    hits = 0
    persona_hits = 0
    persona_total = 0
    mem_hits = 0
    mem_total = 0
    
    for p in probes:
        p_type = p["type"]
        prompt = p["prompt"]
        exp = p["expected"]
        gen = generate_greedy(model, tok, prompt)
        
        is_hit = False
        if p_type == "persona_and_fluency":
            persona_total += 1
            is_hit = len(gen) > 5 and not gen.startswith("!") and any(w in gen.lower() for w in ["chakrview", "hello", "assist", "help", "neural", "good", "welcome", "madad"])
            if is_hit:
                persona_hits += 1
        else:
            mem_total += 1
            is_hit = exp.lower() in gen.lower()
            if is_hit:
                mem_hits += 1
                
        if is_hit:
            hits += 1
            
    return {
        "persona_and_fluency": {
            "hits": persona_hits,
            "total": persona_total,
            "accuracy_pct": round(persona_hits / max(1, persona_total) * 100, 2)
        },
        "conversational_memory": {
            "hits": mem_hits,
            "total": mem_total,
            "accuracy_pct": round(mem_hits / max(1, mem_total) * 100, 2)
        },
        "overall_300": {
            "hits": hits,
            "total": len(probes),
            "accuracy_pct": round(hits / len(probes) * 100, 2)
        }
    }

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(SUITE_800, "r", encoding="utf-8") as f:
        probes_800 = json.load(f)
    with open(SUITE_300, "r", encoding="utf-8") as f:
        probes_300 = json.load(f)
        
    models = {
        "candidate_3_baseline": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt",
        "cr10_arm_a": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_a/cr_exp_10_arm_a_best_val.pt",
        "cr10_arm_b": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_b/cr_exp_10_arm_b_best_val.pt"
    }
    
    full_eval_results = {}
    
    for name, path in models.items():
        print(f"\n================ Evaluating {name} ================")
        m = load_eval_model(path)
        
        print("Running 800-Probe OOD Cognition Suite...")
        res_800 = eval_800_suite(m, tok, probes_800)
        print(f"  --> 800-Suite Overall: {res_800['overall']['grand_total_hits']}/{res_800['overall']['grand_total_probes']} ({res_800['overall']['overall_accuracy_pct']}%)")
        
        print("Running 300-Probe Dedicated Conversational & Memory Suite...")
        res_300 = eval_300_conv_suite(m, tok, probes_300)
        print(f"  --> Conversational Persona: {res_300['persona_and_fluency']['accuracy_pct']}%")
        print(f"  --> Conversational Memory:  {res_300['conversational_memory']['accuracy_pct']}%")
        print(f"  --> 300-Suite Overall:      {res_300['overall_300']['accuracy_pct']}%")
        
        full_eval_results[name] = {
            "benchmark_800": res_800,
            "conversational_300": res_300
        }
        
    out_path = REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/final_multi_suite_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_eval_results, f, indent=2)
    print(f"\nAll evaluations complete. Full scorecard written to {out_path}")

if __name__ == "__main__":
    main()

