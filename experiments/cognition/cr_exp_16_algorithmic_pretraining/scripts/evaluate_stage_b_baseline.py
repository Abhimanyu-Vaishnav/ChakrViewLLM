"""CR-EXP-16 Baseline Evaluator
Evaluates the canonical Stage B checkpoint (hash 96357705a...) on the frozen 1000-probe benchmark.
"""
import sys
import json
import torch
import re
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP16_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining"
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
BENCHMARK_PATH = EXP16_DIR / "benchmark_data/cr16_synthetic_benchmark_1000.json"
OUT_FILE = EXP16_DIR / "benchmark_data/stage_b_baseline_results.json"

def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 25, device: torch.device = torch.device("cpu")) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > 250:
                input_ids = input_ids[:, -250:]
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            
            if next_token in [0, 1, 2]: # pad, eos, bos
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            text = tokenizer.decode(generated, errors="replace")
            if "\n" in text or "Query:" in text or "Record:" in text or "Step " in text:
                break
                
    return tokenizer.decode(generated, errors="replace").strip()

def evaluate_probe(probe, raw_out):
    cat = probe["category"]
    subj = probe.get("subject", "").strip()
    tgt = probe.get("expected_target", "").strip()
    earlier = probe.get("earlier_value", "").strip()
    distractors = probe.get("distractor_values", [])
    
    out_lower = raw_out.lower()
    tgt_lower = tgt.lower()
    
    # Exact sequence copying requires matching the sequence
    if cat == "exact_sequence_copying":
        correct = (tgt_lower in out_lower)
        return {
            "correct": correct,
            "subject_correct": None,
            "value_correct": correct,
            "stale_retained": False,
            "raw_output": raw_out
        }
        
    # Word boundary matching for target and subject
    val_pattern = r'\b' + re.escape(tgt_lower) + r'\b'
    val_match = bool(re.search(val_pattern, out_lower)) if tgt_lower else False
    
    subj_match = False
    if subj:
        subj_pattern = r'\b' + re.escape(subj.lower()) + r'\b'
        subj_match = bool(re.search(subj_pattern, out_lower))
    else:
        subj_match = True
        
    stale_match = False
    if earlier:
        stale_pattern = r'\b' + re.escape(earlier.lower()) + r'\b'
        stale_match = bool(re.search(stale_pattern, out_lower))
        
    dist_match = False
    for d in distractors:
        if d and re.search(r'\b' + re.escape(d.lower()) + r'\b', out_lower):
            dist_match = True
            break
            
    # For single and multi token copying, target is the name
    if "copying" in cat or cat == "subject_identification":
        strict_correct = val_match
    else:
        strict_correct = val_match and (not stale_match)
        
    return {
        "correct": strict_correct,
        "subject_correct": subj_match,
        "value_correct": val_match,
        "stale_retained": stale_match,
        "distractor_retained": dist_match,
        "raw_output": raw_out
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)
        
    print(f"Loaded {len(probes)} benchmark probes.")
    
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512)
    model = ChakrMicro(cfg)
    st = torch.load(STAGE_B_CKPT, map_location="cpu", weights_only=False)
    st_dict = st["model_state_dict"] if "model_state_dict" in st else st
    model.load_state_dict(st_dict, strict=True)
    model.to(device)
    model.eval()
    
    results = []
    cat_stats = {}
    
    for idx, probe in enumerate(probes):
        raw_out = generate_greedy(model, tokenizer, probe["prompt"], max_new_tokens=25, device=device)
        eval_res = evaluate_probe(probe, raw_out)
        eval_res["id"] = probe["id"]
        eval_res["category"] = probe["category"]
        eval_res["prompt"] = probe["prompt"]
        eval_res["expected_target"] = probe["expected_target"]
        results.append(eval_res)
        
        cat = probe["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0, "val_correct": 0, "subj_correct": 0}
        cat_stats[cat]["total"] += 1
        if eval_res["correct"]:
            cat_stats[cat]["correct"] += 1
        if eval_res["value_correct"]:
            cat_stats[cat]["val_correct"] += 1
        if eval_res["subject_correct"]:
            cat_stats[cat]["subj_correct"] += 1
            
        if (idx + 1) % 200 == 0:
            print(f"Evaluated {idx + 1}/{len(probes)} probes...")
            
    summary = {
        "checkpoint": str(STAGE_B_CKPT.name),
        "total_probes": len(probes),
        "overall_correct": sum(1 for r in results if r["correct"]),
        "overall_accuracy": sum(1 for r in results if r["correct"]) / len(probes),
        "categories": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": s["correct"] / s["total"],
                "value_matches": s["val_correct"],
                "subject_matches": s["subj_correct"]
            }
            for cat, s in cat_stats.items()
        }
    }
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "probes": results}, f, indent=2)
        
    print("\n=======================================================")
    print("STAGE B CANONICAL PRETRAINING BASELINE RESULTS:")
    print(f"Overall Accuracy: {summary['overall_accuracy']*100:.2f}% ({summary['overall_correct']}/{len(probes)})")
    for cat, s in summary["categories"].items():
        print(f"  {cat:32}: {s['accuracy']*100:.2f}% ({s['correct']}/{s['total']})")
    print("=======================================================")

if __name__ == "__main__":
    main()

