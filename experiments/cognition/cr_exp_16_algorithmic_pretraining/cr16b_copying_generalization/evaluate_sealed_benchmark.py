"""CR-EXP-16B Sealed Benchmark Evaluator
Evaluates all factorial candidate models and baselines across the 1,500 probes of cr16b_sealed_benchmark_1500.json.
Generates fine-grained breakdowns across all 14 categories (A through N), sequence lengths, templates, and seeds.
"""
import sys
import json
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data"
CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/checkpoints"
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BENCHMARK_PATH = DATA_DIR / "cr16b_sealed_benchmark_1500.json"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Canonical Stage B config
stage_b_raw = torch.load(STAGE_B_CKPT, map_location="cpu", weights_only=False)
config = stage_b_raw["config"]

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_probes = json.load(f)

print(f"Loaded {len(benchmark_probes)} probes from sealed benchmark {BENCHMARK_PATH.name}")

models_to_evaluate = {
    # Baselines
    "stage_b_baseline": STAGE_B_CKPT,
    "cr16a_arm_d": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
    # Factorial Arms Seed 42
    "arm_a_full_single_s42": CKPT_DIR / "arm_a_full_single_seed42/arm_a_full_single_final.pt",
    "arm_b_target_single_s42": CKPT_DIR / "arm_b_target_single_seed42/arm_b_target_single_final.pt",
    "arm_c_full_multi_s42": CKPT_DIR / "arm_c_full_multi_seed42/arm_c_full_multi_final.pt",
    "arm_d_target_multi_s42": CKPT_DIR / "arm_d_target_multi_seed42/arm_d_target_multi_final.pt",
    # Factorial Arms Seed 101
    "arm_a_full_single_s101": CKPT_DIR / "arm_a_full_single_seed101/arm_a_full_single_final.pt",
    "arm_b_target_single_s101": CKPT_DIR / "arm_b_target_single_seed101/arm_b_target_single_final.pt",
    "arm_c_full_multi_s101": CKPT_DIR / "arm_c_full_multi_seed101/arm_c_full_multi_final.pt",
    "arm_d_target_multi_s101": CKPT_DIR / "arm_d_target_multi_seed101/arm_d_target_multi_final.pt",
}

def generate(model, prompt_tokens, max_new_tokens=25):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if token_id == 1:
                break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec:
                break
    out_str = tokenizer.decode(input_ids[0, len(prompt_tokens):].tolist(), errors="replace")
    return out_str.strip().split("\n")[0].strip()

all_model_evals = {}

for model_name, ckpt_path in models_to_evaluate.items():
    print(f"\n--- Evaluating {model_name} on sealed 1,500 benchmark ---")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    cat_stats = {}
    generations = []
    
    total_probes = len(benchmark_probes)
    total_exact = 0
    total_presence = 0
    total_first_correct = 0
    
    for p in benchmark_probes:
        cat = p["category_code"]
        cat_name = p["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {
                "name": cat_name,
                "total": 0,
                "exact": 0,
                "presence": 0,
                "first_token": 0
            }
            
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        target_tokens = tokenizer.encode(target)
        
        # First token check
        with torch.no_grad():
            inp_t = torch.tensor([tokens], dtype=torch.long, device=device)
            out = model(inp_t)
            logits = out[0] if isinstance(out, tuple) else out
            pred_first = logits[0, -1, :].argmax().item()
            first_match = (len(target_tokens) > 0 and pred_first == target_tokens[0])
            
        pred = generate(model, tokens)
        is_exact = (pred == target)
        t_words = set(target.split())
        p_words = set(pred.split())
        is_presence = (len(t_words) > 0 and t_words.issubset(p_words))
        
        cat_stats[cat]["total"] += 1
        if is_exact:
            cat_stats[cat]["exact"] += 1
            total_exact += 1
        if is_presence:
            cat_stats[cat]["presence"] += 1
            total_presence += 1
        if first_match:
            cat_stats[cat]["first_token"] += 1
            total_first_correct += 1
            
        generations.append({
            "id": p["id"],
            "cat": cat,
            "prompt": prompt,
            "target": target,
            "pred": pred,
            "exact": is_exact,
            "presence": is_presence,
            "first_match": first_match
        })
        
    summary = {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "total_probes": total_probes,
        "total_exact": total_exact,
        "exact_accuracy": total_exact / total_probes,
        "total_presence": total_presence,
        "presence_accuracy": total_presence / total_probes,
        "total_first_token": total_first_correct,
        "first_token_accuracy": total_first_correct / total_probes,
        "by_category": cat_stats
    }
    all_model_evals[model_name] = summary
    
    with open(OUT_DIR / f"{model_name}_sealed1500_eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "generations": generations}, f, indent=2)
        
    print(f"[{model_name}] Exact Match: {total_exact}/{total_probes} ({total_exact/total_probes*100:.2f}%) | "
          f"Presence: {total_presence}/{total_probes} ({total_presence/total_probes*100:.2f}%) | "
          f"First Token: {total_first_correct}/{total_probes} ({total_first_correct/total_probes*100:.2f}%)")

with open(OUT_DIR / "cr16b_sealed1500_all_models_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_model_evals, f, indent=2)

print("\nAll 10 models evaluated on sealed 1,500-probe benchmark! Summary saved to eval_outputs/cr16b_sealed1500_all_models_summary.json")

