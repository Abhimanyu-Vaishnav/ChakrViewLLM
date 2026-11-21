import json

with open("experiments/cognition/ucip_06/ucip05_medium_benchmark_eval.json", encoding="utf-8") as f:
    u5_med = json.load(f)["evaluations"]
with open("experiments/cognition/ucip_06/ucip06_medium_candidate_benchmark_eval.json", encoding="utf-8") as f:
    u6_med = json.load(f)["evaluations"]

indices = [0, 60, 110, 150, 210, 260, 310, 360, 410, 450]
for idx in indices:
    p5 = u5_med[idx]
    p6 = u6_med[idx]
    print(f"[{p5['category']}] Input: {p5['prompt']}")
    print(f"  Target Keywords: {p5['expected_keywords']}")
    print(f"  UCIP-05 Output:  '{p5['generated']}' (Passed: {p5['passed']})")
    print(f"  UCIP-06 Output:  '{p6['generated']}' (Passed: {p6['passed']})\n")
