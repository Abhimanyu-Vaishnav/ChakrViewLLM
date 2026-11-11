"""CR-EXP-16C Key-Value Retrieval Evaluation Harness
Evaluates:
- Baseline: Canonical Stage B step 10k
- CR-16A Arm D (Copying checkpoint)
- Retrieval Arm A (Target-Masked) Seed 42 & Seed 101
- Retrieval Arm B (Full-Loss Control) Seed 42 & Seed 101
Across the 1,500 probes of cr16c_sealed_benchmark_1500.json.
Reports exact match, coverage, stale-value errors, distractor captures, and abstention metrics.
"""
import sys
import json
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data"
CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/checkpoints"
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BENCHMARK_PATH = DATA_DIR / "cr16c_sealed_benchmark_1500.json"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

stage_b_raw = torch.load(STAGE_B_CKPT, map_location="cpu", weights_only=False)
config = stage_b_raw["config"]

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_probes = json.load(f)

print(f"Loaded {len(benchmark_probes)} retrieval probes from sealed benchmark.")

models_to_evaluate = {
    "stage_b_baseline": STAGE_B_CKPT,
    "cr16a_arm_d_copying": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
    "retrieval_arm_a_target_s42": CKPT_DIR / "retrieval_arm_a_target_masked_seed42/retrieval_arm_a_target_masked_best.pt",
    "retrieval_arm_a_target_s101": CKPT_DIR / "retrieval_arm_a_target_masked_seed101/retrieval_arm_a_target_masked_best.pt",
    "retrieval_arm_b_full_s42": CKPT_DIR / "retrieval_arm_b_full_loss_seed42/retrieval_arm_b_full_loss_best.pt",
    "retrieval_arm_b_full_s101": CKPT_DIR / "retrieval_arm_b_full_loss_seed101/retrieval_arm_b_full_loss_best.pt",
}

def generate(model, prompt_tokens, max_new_tokens=15):
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

all_eval_summaries = {}

for model_name, ckpt_path in models_to_evaluate.items():
    print(f"\nEvaluating {model_name} on sealed 1,500 retrieval benchmark...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    cat_stats = {}
    generations = []
    
    total_probes = len(benchmark_probes)
    total_exact = 0
    total_stale_errors = 0
    total_distractor_captures = 0
    correct_abstentions = 0
    total_missing_probes = 0
    
    for p in benchmark_probes:
        cat = p["category_code"]
        cat_name = p["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {
                "name": cat_name,
                "total": 0,
                "exact": 0,
                "stale_errors": 0,
                "distractor_captures": 0
            }
            
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        
        pred = generate(model, tokens)
        is_exact = (pred.lower() == target.lower())
        
        # Check stale value error on reassignment (CAT2, CAT3)
        is_stale = False
        if "Update:" in prompt:
            # find original value stored before Update:
            parts = prompt.split("Update:")[0]
            # check if pred is in parts
            if len(pred) > 1 and pred in parts and not is_exact:
                is_stale = True
                total_stale_errors += 1
                cat_stats[cat]["stale_errors"] += 1
                
        # Check distractor capture (CAT8)
        is_distractor = False
        if "Notice:" in prompt:
            notice_part = prompt.split("Notice:")[1].split(".")[0]
            if len(pred) > 1 and pred in notice_part and not is_exact:
                is_distractor = True
                total_distractor_captures += 1
                cat_stats[cat]["distractor_captures"] += 1
                
        if target == "[UNKNOWN]":
            total_missing_probes += 1
            if "[unknown]" in pred.lower() or pred == "":
                correct_abstentions += 1
                
        cat_stats[cat]["total"] += 1
        if is_exact:
            cat_stats[cat]["exact"] += 1
            total_exact += 1
            
        generations.append({
            "id": p["id"],
            "cat": cat,
            "prompt": prompt,
            "target": target,
            "pred": pred,
            "exact": is_exact,
            "stale": is_stale,
            "distractor": is_distractor
        })
        
    abstention_recall = correct_abstentions / max(1, total_missing_probes)
    summary = {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "total_probes": total_probes,
        "total_exact": total_exact,
        "exact_accuracy": total_exact / total_probes,
        "total_stale_errors": total_stale_errors,
        "total_distractor_captures": total_distractor_captures,
        "correct_abstentions": correct_abstentions,
        "total_missing_probes": total_missing_probes,
        "abstention_recall": abstention_recall,
        "by_category": cat_stats
    }
    all_eval_summaries[model_name] = summary
    
    with open(OUT_DIR / f"{model_name}_sealed1500_eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "generations": generations}, f, indent=2)
        
    print(f"[{model_name}] Exact Match: {total_exact}/{total_probes} ({total_exact/total_probes*100:.2f}%) | "
          f"Stale Errors: {total_stale_errors} | Distractor Captures: {total_distractor_captures} | "
          f"Abstention Recall: {correct_abstentions}/{total_missing_probes} ({abstention_recall*100:.1f}%)")

with open(OUT_DIR / "cr16c_sealed1500_all_models_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_eval_summaries, f, indent=2)

print("\nAll retrieval models evaluated on sealed 1,500-probe benchmark!")

