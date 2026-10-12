import json
from pathlib import Path
import torch
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    load_tokenizer_artifacts,
)

tokenizer, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
trained_ckpt = "experiments/batch63_66_training_generation_debug/learning_experiment/run_50steps/sft_checkpoint_step_50.pt"

evaluator = DeterministicEvaluator(
    checkpoint_path=trained_ckpt,
    tokenizer_dir="release/chakrview-v0.5.0/data/experiments/vocab_4096",
    strip_prompt_delimiters=False,  # EXACT delimiter matching
)

def evaluate_split(split_path: Path, name: str):
    with open(split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    evaluated = []
    exact_match_count = 0
    eos_count = 0
    repetition_count = 0

    print(f"--- Evaluating {name} ({len(records)} records) ---")
    for r in records:
        prompt_tokens = [0] + tokenizer.encode(r["prompt"])
        gen = list(prompt_tokens)
        eos_reached = False

        for _ in range(25):
            x = torch.tensor([gen], dtype=torch.long)
            logits = evaluator.model(x)
            nxt = torch.argmax(logits[0, -1, :]).item()
            gen.append(nxt)
            if nxt == 1:
                eos_reached = True
                break

        new_tokens = gen[len(prompt_tokens):]
        decoded = tokenizer.decode(new_tokens)
        clean_gen = decoded.replace("<|im_end|>", "").strip()
        expected = r["expected_answer"].strip()

        exact = (clean_gen == expected)
        if exact:
            exact_match_count += 1
        if eos_reached:
            eos_count += 1

        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        rep = (len(lines) > 2 and len(set(lines)) <= len(lines) // 2)
        if rep:
            repetition_count += 1

        evaluated.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "expected_answer": expected,
            "generated_text": decoded,
            "clean_generated": clean_gen,
            "exact_match": exact,
            "eos_reached": eos_reached,
            "repetition": rep,
            "token_count": len(new_tokens),
        })

    dataset_res = evaluator.evaluate_dataset(split_path)

    return {
        "split_name": name,
        "total_records": len(records),
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": round(exact_match_count / len(records), 4),
        "eos_count": eos_count,
        "eos_rate": round(eos_count / len(records), 4),
        "repetition_count": repetition_count,
        "repetition_rate": round(repetition_count / len(records), 4),
        "token_loss": dataset_res.get("mean_token_loss"),
        "perplexity": dataset_res.get("perplexity"),
        "active_tokens": dataset_res.get("total_active_tokens"),
        "records": evaluated,
    }

data_dir = Path("experiments/batch59_62_learning_verification/data")
train_res = evaluate_split(data_dir / "train_synthetic.jsonl", "TRAIN")
val_res = evaluate_split(data_dir / "val_synthetic.jsonl", "VALIDATION")
bench_res = evaluate_split(data_dir / "benchmark_synthetic.jsonl", "HELD_OUT_BENCHMARK")

summary = {
    "checkpoint": trained_ckpt,
    "splits": {
        "TRAIN": train_res,
        "VALIDATION": val_res,
        "HELD_OUT_BENCHMARK": bench_res,
    }
}

out_path = Path("experiments/batch63_66_training_generation_debug/learning_experiment/run_50steps/evaluation_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nEvaluation summary saved to {out_path}")
print(f"TRAIN Exact Match: {train_res['exact_match_count']}/{train_res['total_records']} ({train_res['exact_match_accuracy']}) | Loss: {train_res['token_loss']} | EOS: {train_res['eos_rate']}")
print(f"VAL Exact Match:   {val_res['exact_match_count']}/{val_res['total_records']} ({val_res['exact_match_accuracy']}) | Loss: {val_res['token_loss']} | EOS: {val_res['eos_rate']}")
print(f"BENCH Exact Match: {bench_res['exact_match_count']}/{bench_res['total_records']} ({bench_res['exact_match_accuracy']}) | Loss: {bench_res['token_loss']} | EOS: {bench_res['eos_rate']}")
