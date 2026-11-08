"""
CR-EXP-13 Section 3: Independent Reproduction of All Five Model Evaluations
Evaluates:
1. CR-10 Arm B
2. CR-11 Candidate A
3. CR-11 Candidate B
4. CR-12 Candidate A
5. CR-12 Candidate B

Uses the exact frozen CR-12 benchmark (1,500 probes, SHA-256: 19565749...).
Saves raw predictions, correctness, abstention flags, and full metadata.
Outputs:
- cr_exp_13_reproduced_evaluations.json
- Individual raw prediction logs per model
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
BENCHMARK_1500 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_12_conversational_consolidation" / "benchmark_data" / "benchmark_1500_suite.json"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_13_independent_verification" / "audit_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            if nxt == 13 and len(curr) - len(enc) > 3:
                break
    gen_ids = curr[len(enc):]
    try:
        return tok.decode(gen_ids, errors="replace").strip()
    except Exception:
        return "".join([tok.decode([tid], errors="replace") for tid in gen_ids]).strip()

def evaluate_model_on_suite(model, tok, probes):
    cat_scores = {}
    abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state", "unmentioned", "not mention"]
    detailed_results = []
    
    total_valid_outputs = 0
    total_invalid_outputs = 0
    total_abstained = 0
    
    ans_correct = 0
    ans_total = 0
    unans_correct_abstain = 0
    unans_total = 0
    false_abstain_count = 0
    
    for p in probes:
        pid = p["id"]
        cat = p["category"]
        prompt = p["prompt"]
        target = p.get("expected_target", "")
        
        gen = generate_greedy(model, tok, prompt)
        is_abstained = any(cue in gen.lower() for cue in abstain_cues)
        if is_abstained:
            total_abstained += 1
            
        is_hit = False
        is_hallucinated = False
        is_invalid = len(gen.strip()) == 0 or gen.startswith("!")
        
        if is_invalid:
            total_invalid_outputs += 1
        else:
            total_valid_outputs += 1
            
        if cat == "natural_conversation":
            is_hit = len(gen) > 5 and not gen.startswith("!") and any(w in gen.lower() for w in ["chakrview", "hello", "assist", "help", "neural", "good", "welcome", "madad"])
        elif cat == "single_hop_retrieval":
            is_hit = target.lower() in gen.lower()
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "multi_entity_distractor":
            distractors = p.get("distractor_values", [])
            has_tgt = target.lower() in gen.lower()
            has_dist = any(d.lower() in gen.lower() for d in distractors)
            is_hit = has_tgt and not has_dist
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "multiturn_working_memory":
            earlier = p.get("earlier_value", "")
            has_latest = target.lower() in gen.lower()
            has_earlier = earlier.lower() in gen.lower() if earlier else False
            is_hit = has_latest and not has_earlier
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "counterfactual_updates":
            earlier = p.get("earlier_value", "")
            has_latest = target.lower() in gen.lower()
            has_earlier = earlier.lower() in gen.lower() if earlier else False
            is_hit = has_latest and not has_earlier
            if not is_hit and not is_abstained and len(gen) > 3:
                is_hallucinated = True
        elif cat == "answerability_calibration":
            is_ans = p.get("answerable", True)
            if is_ans:
                ans_total += 1
                is_hit = target.lower() in gen.lower()
                if is_hit:
                    ans_correct += 1
                if is_abstained:
                    false_abstain_count += 1
            else:
                unans_total += 1
                is_hit = is_abstained
                if is_hit:
                    unans_correct_abstain += 1
                elif len(gen) > 3 and not gen.startswith("!"):
                    is_hallucinated = True
        elif cat == "conditional_logic":
            is_hit = gen.lower().startswith(target.lower())
        elif cat == "elementary_arithmetic":
            is_hit = target in gen
        elif cat == "constraint_satisfaction":
            is_hit = target.lower() in gen.lower()
            
        cat_scores.setdefault(cat, {"hits": 0, "total": 0, "incorrect": 0})
        cat_scores[cat]["total"] += 1
        if is_hit:
            cat_scores[cat]["hits"] += 1
        else:
            cat_scores[cat]["incorrect"] += 1
            
        detailed_results.append({
            "probe_id": pid,
            "category": cat,
            "prompt": prompt,
            "expected_target": target,
            "generated": gen,
            "is_correct": is_hit,
            "is_abstained": is_abstained,
            "is_hallucinated": is_hallucinated,
            "is_invalid": is_invalid
        })
        
    summary = {}
    tot_hits = sum(d["hits"] for d in cat_scores.values())
    tot_samples = sum(d["total"] for d in cat_scores.values())
    for cat, d in sorted(cat_scores.items()):
        summary[cat] = {
            "hits": d["hits"],
            "incorrect": d["incorrect"],
            "total": d["total"],
            "accuracy_pct": round(d["hits"] / d["total"] * 100, 2)
        }
    summary["overall"] = {
        "grand_total_hits": tot_hits,
        "grand_total_incorrect": tot_samples - tot_hits,
        "grand_total_probes": tot_samples,
        "overall_accuracy_pct": round(tot_hits / tot_samples * 100, 2),
        "total_valid_outputs": total_valid_outputs,
        "total_invalid_outputs": total_invalid_outputs,
        "total_abstained": total_abstained,
        "answerable_accuracy_pct": round(ans_correct / max(1, ans_total) * 100, 2),
        "unanswerable_abstention_recall_pct": round(unans_correct_abstain / max(1, unans_total) * 100, 2),
        "false_abstention_rate_pct": round(false_abstain_count / max(1, ans_total) * 100, 2)
    }
    return summary, detailed_results

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(BENCHMARK_1500, "r", encoding="utf-8") as f:
        probes_1500 = json.load(f)
        
    models = {
        "cr10_arm_b": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_b/cr_exp_10_arm_b_best_val.pt",
            "hash": "8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488"
        },
        "cr11_candidate_a": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_a/cr_exp_11_candidate_a_best_val.pt",
            "hash": "4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5"
        },
        "cr11_candidate_b": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_b/cr_exp_11_candidate_b_best_val.pt",
            "hash": "b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce"
        },
        "cr12_candidate_a": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_a/cr_exp_12_candidate_a_best_val.pt",
            "hash": "cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2"
        },
        "cr12_candidate_b": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_b/cr_exp_12_candidate_b_best_val.pt",
            "hash": "60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e"
        }
    }
    
    reproduced_metrics = {}
    
    for name, info in models.items():
        print(f"\n[CR-EXP-13 Independent Verification] Running {name}...")
        t0 = time.time()
        model = load_eval_model(info["path"])
        summary, details = evaluate_model_on_suite(model, tok, probes_1500)
        dt = time.time() - t0
        print(f"[{name}] Completed in {dt:.1f}s. Accuracy: {summary['overall']['overall_accuracy_pct']}% ({summary['overall']['grand_total_hits']}/1500)")
        
        reproduced_metrics[name] = {
            "checkpoint_hash": info["hash"],
            "summary": summary
        }
        
        # Save individual raw predictions
        det_path = OUT_DIR / f"{name}_reproduced_detailed.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)
            
    out_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_13_independent_verification" / "REPRODUCED_METRICS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reproduced_metrics, f, indent=2)
    print(f"\nVerification metrics saved to {out_file}")

if __name__ == "__main__":
    main()

