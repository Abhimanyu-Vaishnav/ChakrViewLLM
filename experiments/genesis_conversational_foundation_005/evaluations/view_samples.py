import json

with open("experiments/genesis_conversational_foundation_005/evaluations/100_prompt_benchmark_results.json", encoding="utf-8") as f:
    bench = json.load(f)

for model_name, data in bench.items():
    print("=" * 60)
    print("MODEL:", model_name)
    evals = data["evaluations"]
    for idx in [0, 2, 5, 20, 25, 40, 45, 60, 75, 95]:
        item = evals[idx]
        p = item["prompt"].split("\n")[0]
        r = item["response"]
        cat = item["category"]
        lang = item["language"]
        print(f"[{cat}|{lang}] {p}\n  --> {repr(r)}\n")
