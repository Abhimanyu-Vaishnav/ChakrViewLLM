import sys, json, torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BENCHMARK_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
OUTPUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/eval_outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Canonical Stage B config
stage_b_raw = torch.load(
    REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    map_location="cpu",
    weights_only=False
)
config = stage_b_raw["config"]

# Checkpoints to re-evaluate
checkpoints = {
    "stage_b_baseline": REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    "stage_a_pilot": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/checkpoints/pilot/stage_a_pilot_final.pt",
    "cr16a_arm_a": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_a_full_seq/arm_a_full_seq_final.pt",
    "cr16a_arm_b": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_b_target_masked/arm_b_target_masked_final.pt",
    "cr16a_arm_c": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_c_curriculum/arm_c_curriculum_best.pt",
    "cr16a_arm_d": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
}

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    bench = json.load(f)

copy_probes = [p for p in bench if p["category"] in ["exact_sequence_copying", "single_token_copying", "multi_token_copying"]]
print(f"Loaded {len(copy_probes)} copying probes from original frozen benchmark.")

def generate(model, prompt_tokens, max_new_tokens=20):
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

for name, ckpt_path in checkpoints.items():
    print(f"\nEvaluating model: {name} from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    results = {}
    generations = []
    
    for p in copy_probes:
        cat = p["category"]
        if cat not in results:
            results[cat] = {"total": 0, "exact": 0, "token_presence": 0, "first_token_correct": 0}
            
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        target_tokens = tokenizer.encode(target)
        
        # Check first token
        with torch.no_grad():
            inp_t = torch.tensor([tokens], dtype=torch.long, device=device)
            out_logits = model(inp_t)
            logits_first = out_logits[0] if isinstance(out_logits, tuple) else out_logits
            pred_first_tok = logits_first[0, -1, :].argmax().item()
            first_match = (len(target_tokens) > 0 and pred_first_tok == target_tokens[0])
            
        pred = generate(model, tokens)
        is_exact = (pred == target)
        t_words = set(target.split())
        p_words = set(pred.split())
        is_presence = (len(t_words) > 0 and t_words.issubset(p_words))
        
        results[cat]["total"] += 1
        if is_exact:
            results[cat]["exact"] += 1
        if is_presence:
            results[cat]["token_presence"] += 1
        if first_match:
            results[cat]["first_token_correct"] += 1
            
        generations.append({
            "id": p["id"],
            "category": cat,
            "prompt": prompt,
            "target": target,
            "pred": pred,
            "is_exact": is_exact,
            "is_presence": is_presence,
            "first_match": first_match
        })
        
    summary = {
        "model_name": name,
        "checkpoint_path": str(ckpt_path),
        "total_copy_probes": len(copy_probes),
        "total_exact": sum(r["exact"] for r in results.values()),
        "exact_accuracy": sum(r["exact"] for r in results.values()) / len(copy_probes),
        "total_presence": sum(r["token_presence"] for r in results.values()),
        "presence_accuracy": sum(r["token_presence"] for r in results.values()) / len(copy_probes),
        "by_category": results
    }
    all_eval_summaries[name] = summary
    
    # Save individual raw predictions
    with open(OUTPUT_DIR / f"{name}_reproduced_eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "generations": generations}, f, indent=2)
        
    print(f"[{name}] Exact: {summary['total_exact']}/{summary['total_copy_probes']} ({summary['exact_accuracy']*100:.2f}%) | "
          f"Exact Seq Copying: {results['exact_sequence_copying']['exact']}/{results['exact_sequence_copying']['total']} | "
          f"Single Name: {results['single_token_copying']['exact']}/{results['single_token_copying']['total']} | "
          f"Multi Name: {results['multi_token_copying']['exact']}/{results['multi_token_copying']['total']}")

with open(OUTPUT_DIR / "all_models_reproduction_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_eval_summaries, f, indent=2)

print("\nReproduction evaluation complete! Summaries saved to eval_outputs/all_models_reproduction_summary.json")

