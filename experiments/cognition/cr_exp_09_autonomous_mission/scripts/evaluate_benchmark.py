"""
CR-EXP-09 Comprehensive Benchmark Evaluator
Evaluates any model checkpoint across the 600-probe benchmark suite:
- Conversational Understanding (75)
- Single-Hop & Counterfactual Binding (75)
- Multi-Entity Distractor Disambiguation (75)
- Missing-Information Abstention (75)
- Multi-Turn State Tracking (75)
- Structured Arithmetic State (75)
- Conditional Logical Reasoning (75)
- Constraint Satisfaction (75)

Outputs detailed per-probe results and category-level scorecard.
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
BENCHMARK_FILE = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission" / "benchmark" / "benchmark_suite.json"

def load_eval_model(ckpt_path):
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

def generate_greedy(model, tok, prompt_text, max_new_tokens=20):
    enc = tok.encode(prompt_text)
    curr = list(enc)
    for _ in range(max_new_tokens):
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1]
            nxt = logits.argmax().item()
            curr.append(nxt)
            if nxt in [13] and len(curr) - len(enc) > 5:
                break
    gen_ids = curr[len(enc):]
    try:
        return tok.decode(gen_ids, errors="replace").strip()
    except Exception:
        # Fallback to byte replacement
        return "".join([tok.decode([tid], errors="replace") for tid in gen_ids]).strip()

def evaluate_checkpoint(ckpt_path, model_name="model"):
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = load_eval_model(ckpt_path)
    
    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        probes = json.load(f)
        
    category_scores = {}
    detailed_outputs = []
    
    for p in probes:
        cat = p["category"]
        prompt = p["prompt"]
        target = p["expected_target"]
        
        gen = generate_greedy(model, tok, prompt, max_new_tokens=15)
        
        # Category-specific success criteria
        is_correct = False
        if cat == "conversational_understanding":
            # Coherent persona response check
            is_correct = len(gen) > 5 and not gen.startswith("!") and any(w in gen.lower() for w in ["chakrview", "hello", "assist", "help", "neural", "good", "welcome", "madad"])
        elif cat in ["single_hop_and_counterfactual", "constraint_satisfaction"]:
            is_correct = target.lower() in gen.lower()
        elif cat == "multi_entity_distractor":
            # Must contain target city AND NOT contain distractor city
            d_val = p.get("distractor_value", "")
            contains_target = target.lower() in gen.lower()
            contains_distractor = d_val.lower() in gen.lower() if d_val else False
            is_correct = contains_target and not contains_distractor
        elif cat == "missing_info_abstention":
            # Must express uncertainty / lack of info
            abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state"]
            is_correct = any(cue in gen.lower() for cue in abstain_cues)
        elif cat == "multiturn_state_tracking":
            # Must state the latest city (target) and not the earlier city
            earlier = p.get("earlier_value", "")
            contains_latest = target.lower() in gen.lower()
            contains_earlier = earlier.lower() in gen.lower() if earlier else False
            is_correct = contains_latest and not contains_earlier
        elif cat == "structured_arithmetic_state":
            # Exact numerical token match
            is_correct = target in gen
        elif cat == "conditional_logical_reasoning":
            # Boolean answer match
            is_correct = gen.lower().startswith(target.lower())
            
        category_scores.setdefault(cat, {"hits": 0, "total": 0})
        category_scores[cat]["total"] += 1
        if is_correct:
            category_scores[cat]["hits"] += 1
            
        detailed_outputs.append({
            "id": p["id"],
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated": gen,
            "is_correct": is_correct
        })
        
    summary = {}
    total_hits = 0
    total_samples = 0
    for cat, d in category_scores.items():
        acc = round(d["hits"] / d["total"] * 100, 2)
        summary[cat] = {
            "hits": d["hits"],
            "total": d["total"],
            "accuracy_pct": acc
        }
        total_hits += d["hits"]
        total_samples += d["total"]
        
    summary["overall"] = {
        "grand_total_hits": total_hits,
        "grand_total_probes": total_samples,
        "overall_accuracy_pct": round(total_hits / total_samples * 100, 2)
    }
    
    return summary, detailed_outputs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_benchmark.py <ckpt_path> [out_json]")
        sys.exit(1)
    ckpt = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else "eval_result.json"
    summary, outputs = evaluate_checkpoint(ckpt)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": outputs}, f, indent=2)
    print(f"Evaluation complete. Summary saved to {out_json}")
    for cat, vals in summary.items():
        if cat != "overall":
            print(f"  {cat:32s}: {vals['hits']:2d}/{vals['total']} ({vals['accuracy_pct']:5.2f}%)")
    print(f"OVERALL: {summary['overall']['grand_total_hits']}/{summary['overall']['grand_total_probes']} ({summary['overall']['overall_accuracy_pct']}%)")
