"""CR-EXP-16 Stage A Pilot Evaluator
Evaluates the Stage A pilot checkpoint on the first 300 copying probes of the benchmark:
- Exact Sequence Copying (100)
- Single-Token Copying (100)
- Multi-Token Copying (100)
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
PILOT_CKPT = EXP16_DIR / "checkpoints/pilot/stage_a_pilot_best.pt"
BENCHMARK_PATH = EXP16_DIR / "benchmark_data/cr16_synthetic_benchmark_1000.json"
OUT_FILE = EXP16_DIR / "benchmark_data/stage_a_pilot_eval.json"

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
            
            if next_token in [0, 1, 2]:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            text = tokenizer.decode(generated, errors="replace")
            if "\n" in text or "Query:" in text or "Record:" in text:
                break
                
    return tokenizer.decode(generated, errors="replace").strip()

def evaluate_copy_probe(probe, raw_out):
    tgt = probe["expected_target"].strip().lower()
    clean_out = raw_out.strip().split("\n")[0].strip().lower()
    
    # Exact sequence match
    exact_match = (tgt == clean_out)
    
    # Word-boundary presence
    val_pattern = r'\b' + re.escape(tgt) + r'\b'
    val_present = bool(re.search(val_pattern, clean_out))
    
    return {
        "exact_match": exact_match,
        "value_present": val_present,
        "raw_output": raw_out
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)
        
    copy_probes = [p for p in probes if "copying" in p["category"]]
    print(f"Loaded {len(copy_probes)} copying probes for pilot evaluation.")
    
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512)
    model = ChakrMicro(cfg)
    st = torch.load(PILOT_CKPT, map_location="cpu")
    st_dict = st["model_state_dict"] if "model_state_dict" in st else st
    model.load_state_dict(st_dict, strict=True)
    model.to(device)
    model.eval()
    
    cat_stats = {}
    results = []
    
    for probe in copy_probes:
        raw_out = generate_greedy(model, tokenizer, probe["prompt"], max_new_tokens=25, device=device)
        res = evaluate_copy_probe(probe, raw_out)
        res["id"] = probe["id"]
        res["category"] = probe["category"]
        res["prompt"] = probe["prompt"]
        res["expected_target"] = probe["expected_target"]
        results.append(res)
        
        cat = probe["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "exact": 0, "present": 0}
        cat_stats[cat]["total"] += 1
        if res["exact_match"]:
            cat_stats[cat]["exact"] += 1
        if res["value_present"]:
            cat_stats[cat]["present"] += 1
            
    summary = {
        "checkpoint": str(PILOT_CKPT.name),
        "total_probes": len(copy_probes),
        "total_exact": sum(1 for r in results if r["exact_match"]),
        "exact_accuracy": sum(1 for r in results if r["exact_match"]) / len(copy_probes),
        "categories": {
            cat: {
                "total": s["total"],
                "exact": s["exact"],
                "exact_acc": s["exact"] / s["total"],
                "value_present": s["present"],
                "present_acc": s["present"] / s["total"]
            }
            for cat, s in cat_stats.items()
        }
    }
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "probes": results}, f, indent=2)
        
    print("\n=======================================================")
    print("STAGE A PILOT EVALUATION RESULTS:")
    print(f"Overall Copying Exact Accuracy: {summary['exact_accuracy']*100:.2f}% ({summary['total_exact']}/{len(copy_probes)})")
    for cat, s in summary["categories"].items():
        print(f"  {cat:28}: Exact={s['exact_acc']*100:.2f}% ({s['exact']}/{s['total']}), Present={s['present_acc']*100:.2f}% ({s['value_present']}/{s['total']})")
    print("=======================================================")

if __name__ == "__main__":
    main()

