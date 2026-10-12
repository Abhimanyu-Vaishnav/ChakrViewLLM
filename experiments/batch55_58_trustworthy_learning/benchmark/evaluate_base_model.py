import json
from pathlib import Path
import torch
from experiments.batch55_58_trustworthy_learning.benchmark.micro_benchmark import (
    VERIFIED_MICROBENCHMARK_ITEMS,
    evaluate_benchmark_item,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    load_tokenizer_artifacts,
)

tokenizer, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
base_ckpt = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

evaluator = DeterministicEvaluator(
    checkpoint_path=base_ckpt,
    tokenizer_dir="release/chakrview-v0.5.0/data/experiments/vocab_4096",
)

results = []
print("Evaluating canonical Step 10,000 checkpoint on 9 verified micro-benchmark items:")
for item in VERIFIED_MICROBENCHMARK_ITEMS:
    prompt_tokens = [0] + tokenizer.encode(item.prompt)
    gen = list(prompt_tokens)
    eos_reached = False

    for _ in range(40):
        x = torch.tensor([gen], dtype=torch.long)
        logits = evaluator.model(x)
        nxt = torch.argmax(logits[0, -1, :]).item()
        gen.append(nxt)
        if nxt == 1:
            eos_reached = True
            break

    new_tokens = gen[len(prompt_tokens):]
    decoded_text = tokenizer.decode(new_tokens)
    item_res = evaluate_benchmark_item(item, decoded_text, eos_reached)
    item_res["generated_tokens_count"] = len(new_tokens)
    item_res["generated_token_ids"] = new_tokens
    results.append(item_res)
    succ = item_res["task_success"]
    rep = item_res["repetition_detected"]
    preview = repr(decoded_text[:35])
    print(f"  [{item.domain:11s}] {item.id}: eos={eos_reached}, success={succ}, rep={rep}, output={preview}")

summary = {
    "checkpoint": base_ckpt,
    "total_items": len(results),
    "successful_items": sum(1 for r in results if r["task_success"]),
    "eos_rate": sum(1 for r in results if r["eos_reached"]) / len(results),
    "repetition_rate": sum(1 for r in results if r["repetition_detected"]) / len(results),
    "per_domain_success": {
        d: sum(1 for r in results if r["domain"] == d and r["task_success"])
        for d in ["mathematics", "code", "science"]
    },
    "items": results,
}

out_path = Path("experiments/batch55_58_trustworthy_learning/benchmark/base_model_benchmark_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Base model evaluation complete! Summary:")
print("  Success:", summary["successful_items"], "/", summary["total_items"])
print("  EOS Rate:", f"{summary['eos_rate']*100:.1f}%")
print("  Repetition Rate:", f"{summary['repetition_rate']*100:.1f}%")
