import json

p = 'experiments/ucip11/results/full_benchmark_evaluation_results.json'
d = json.load(open(p))
for cond, data in d.items():
    print(f"=== {cond} ===")
    print("Dev:", data['dev_results'])
    print("Sealed:", data['sealed_results'])
    passed_items = [e for e in data['sealed_evaluations'] if e['passed']]
    for pi in passed_items:
        print(f"  Passed: {pi['id']} | Prompt: {pi['prompt'][:40]} | Out: {pi['raw_output']} | Reason: {pi['reason']}")
    print("-" * 50)
