import sys, json, torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(str(REPO_ROOT / "data" / "experiments" / "vocab_4096"))
device = "cuda" if torch.cuda.is_available() else "cpu"

stage_b = torch.load("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt", map_location="cpu", weights_only=False)
config = stage_b["config"]

ckpt_path = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_c_curriculum/arm_c_curriculum_best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = ChakrMicro(config).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open(REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json") as f:
    bench = json.load(f)

copy_probes = [p for p in bench if p["category"] in ["exact_sequence_copying", "single_token_copying", "multi_token_copying"]]

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

results = {}
generations = []
for p in copy_probes:
    cat = p["category"]
    if cat not in results:
        results[cat] = {"total": 0, "exact": 0, "token_presence": 0}
    prompt = p["prompt"]
    target = p["expected_target"].strip()
    tokens = tokenizer.encode(prompt)
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
        
    generations.append({
        "id": p["id"],
        "category": cat,
        "prompt": prompt,
        "target": target,
        "pred": pred,
        "is_exact": is_exact,
        "is_presence": is_presence
    })

print("=== ARM C BENCHMARK COPYING RESULTS ===")
total_exact = sum(r["exact"] for r in results.values())
total_probes = sum(r["total"] for r in results.values())
total_presence = sum(r["token_presence"] for r in results.values())
for cat, res in results.items():
    print(f"{cat}: Exact: {res['exact']}/{res['total']} ({res['exact']/res['total']*100:.1f}%), Token Presence: {res['token_presence']}/{res['total']} ({res['token_presence']/res['total']*100:.1f}%)")
print(f"OVERALL: Exact: {total_exact}/{total_probes} ({total_exact/total_probes*100:.2f}%), Token Presence: {total_presence}/{total_probes} ({total_presence/total_probes*100:.2f}%)")

print("\nSample Generations:")
for gen in generations[::30][:10]:
    print(f"[{gen['category']}] P: {repr(gen['prompt'].splitlines()[0])} -> Pred: {repr(gen['pred'])} | Target: {repr(gen['target'])} | Match: {gen['is_exact']}")

eval_output_path = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/arm_c_eval_results.json"
with open(eval_output_path, "w") as f:
    json.dump({
        "summary": {
            "total_probes": total_probes,
            "total_exact": total_exact,
            "exact_accuracy": total_exact / total_probes,
            "total_presence": total_presence,
            "presence_accuracy": total_presence / total_probes,
            "by_category": results
        },
        "generations": generations
    }, f, indent=2)
print(f"Saved evaluation results to {eval_output_path}")

