"""
CR-EXP-11 Phase 1: Independent Reproduction and Diagnostic Audit of CR-EXP-10
Evaluates Cand 3, CR-10 Arm A, and CR-10 Arm B on the frozen 800-probe and 300-probe suites.
Records per-probe outputs: probe_id, category, expected, generated, normalized, correctness, abstention, hallucination.
"""

import sys
import json
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
SUITE_800 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_10_readiness" / "benchmark_data" / "benchmark_800_suite.json"
SUITE_300 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_10_readiness" / "benchmark_data" / "conversational_300_suite.json"

OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"

def load_eval_model(ckpt_path):
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    m.eval()
    return m

def generate_greedy(model, tok, prompt, max_new_tokens=25):
    enc = tok.encode(prompt)
    curr = list(enc)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([curr], dtype=torch.long)
            logits = model(x)[0, -1]
            nxt = logits.argmax().item()
            curr.append(nxt)
            if nxt == 13 and len(curr) - len(enc) > 5:  # stop on newline after at least 5 tokens
                break
    gen_ids = curr[len(enc):]
    try:
        return tok.decode(gen_ids, errors="replace").strip()
    except Exception:
        return "".join([tok.decode([tid], errors="replace") for tid in gen_ids]).strip()

def run_800_eval(model, tok, probes, ckpt_hash):
    cat_scores = {}
    detailed_results = []
    abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state", "unmentioned", "not mention"]
    
    for p in probes:
        pid = p["id"]
        cat = p["category"]
        prompt = p["prompt"]
        target = p["expected_target"]
        gen = generate_greedy(model, tok, prompt)
        
        is_hit = False
        is_abstained = any(cue in gen.lower() for cue in abstain_cues)
        is_hallucinated = False
        
        if cat == "conversational_understanding":
            is_hit = len(gen) > 5 and not gen.startswith("!") and any(w in gen.lower() for w in ["chakrview", "hello", "assist", "help", "neural", "good", "welcome", "madad"])
        elif cat in ["single_hop_and_counterfactual", "constraint_satisfaction"]:
            is_hit = target.lower() in gen.lower()
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "multi_entity_distractor":
            d_val = p.get("distractor_value", "")
            contains_target = target.lower() in gen.lower()
            contains_distractor = d_val.lower() in gen.lower() if d_val else False
            is_hit = contains_target and not contains_distractor
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "missing_info_abstention":
            is_hit = is_abstained
            if not is_hit and len(gen) > 3 and not gen.startswith("!"):
                is_hallucinated = True
        elif cat == "multiturn_state_tracking":
            earlier = p.get("earlier_value", "")
            contains_latest = target.lower() in gen.lower()
            contains_earlier = earlier.lower() in gen.lower() if earlier else False
            is_hit = contains_latest and not contains_earlier
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "structured_arithmetic_state":
            is_hit = target in gen
        elif cat == "conditional_logical_reasoning":
            is_hit = gen.lower().startswith(target.lower())
            
        cat_scores.setdefault(cat, {"hits": 0, "total": 0})
        cat_scores[cat]["total"] += 1
        if is_hit:
            cat_scores[cat]["hits"] += 1
            
        detailed_results.append({
            "probe_id": pid,
            "category": cat,
            "prompt": prompt,
            "expected_target": target,
            "generated": gen,
            "is_correct": is_hit,
            "is_abstained": is_abstained,
            "is_hallucinated": is_hallucinated,
            "checkpoint_hash": ckpt_hash
        })
        
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
    return summary, detailed_results

def run_300_eval(model, tok, probes, ckpt_hash):
    hits = 0
    persona_hits = 0
    persona_total = 0
    mem_hits = 0
    mem_total = 0
    detailed_results = []
    
    for p in probes:
        pid = p["id"]
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
            
        detailed_results.append({
            "probe_id": pid,
            "type": p_type,
            "prompt": prompt,
            "expected": exp,
            "generated": gen,
            "is_correct": is_hit,
            "checkpoint_hash": ckpt_hash
        })
        
    summary = {
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
    return summary, detailed_results

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(SUITE_800, "r", encoding="utf-8") as f:
        probes_800 = json.load(f)
    with open(SUITE_300, "r", encoding="utf-8") as f:
        probes_300 = json.load(f)
        
    models = {
        "candidate_3_baseline": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt",
            "hash": "de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592"
        },
        "cr10_arm_a": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_a/cr_exp_10_arm_a_best_val.pt",
            "hash": "dc51e4ab338006ff7380bb7500963a17a8740b9b5eab75bb452166463892e74c"
        },
        "cr10_arm_b": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_arm_b/cr_exp_10_arm_b_best_val_copy.pt",
            "hash": "8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488"
        }
    }
    
    all_reproduced_metrics = {}
    
    for name, info in models.items():
        print(f"\n--- Reproducing: {name} ---")
        t0 = time.time()
        model = load_eval_model(info["path"])
        sum_800, det_800 = run_800_eval(model, tok, probes_800, info["hash"])
        sum_300, det_300 = run_300_eval(model, tok, probes_300, info["hash"])
        dt = time.time() - t0
        print(f"Done in {dt:.1f}s. 800 Overall: {sum_800['overall']['overall_accuracy_pct']}%, 300 Overall: {sum_300['overall_300']['accuracy_pct']}%")
        
        all_reproduced_metrics[name] = {
            "checkpoint_hash": info["hash"],
            "benchmark_800": sum_800,
            "conversational_300": sum_300
        }
        
        # Save detailed logs
        det_path = OUT_DIR / f"{name}_detailed_probes.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump({"probes_800": det_800, "probes_300": det_300}, f, indent=2)
            
    out_metrics_path = OUT_DIR / "cr_exp_10_reproduced_metrics.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_reproduced_metrics, f, indent=2)
    print(f"\nSaved reproduction metrics to {out_metrics_path}")

if __name__ == "__main__":
    main()

