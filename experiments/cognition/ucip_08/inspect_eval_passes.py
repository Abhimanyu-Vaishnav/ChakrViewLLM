import json

with open("D:/Project/ChakrView/experiments/cognition/ucip_07/eval_ucip07_medium.json", "r", encoding="utf-8") as f:
    d = json.load(f)

passed_samples = [p for p in d["evaluations"] if p["passed"]][:40]
print(f"Total detailed: {len(d['evaluations'])}, Total passed: {len([p for p in d['evaluations'] if p['passed']])}")
for s in passed_samples:
    print("=" * 60)
    print(f"ID: {s['id']} | Cat: {s['category']}")
    print(f"Prompt: {s['prompt']}")
    print(f"Expected KW: {s.get('expected_keywords')}")
    print(f"Generated: \"{s['generated']}\"")
    print(f"Details: {s['details']}")
