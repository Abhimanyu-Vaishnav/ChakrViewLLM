import json

models = [
    ('UCIP-06 Medium', 'experiments/cognition/ucip_08/eval_ucip06_medium_owner_reality.json'),
    ('UCIP-07 Medium', 'experiments/cognition/ucip_08/eval_ucip07_medium_owner_reality.json'),
    ('UCIP-08 Medium', 'experiments/cognition/ucip_08/eval_ucip08_medium_owner_reality.json')
]

loaded = [(name, json.load(open(path, encoding='utf-8'))) for name, path in models]
cats = list(loaded[0][1]['categories'].keys())

print(f"{'Category':<25} | {'UCIP-06 Medium':<16} | {'UCIP-07 Medium':<16} | {'UCIP-08 Candidate':<16}")
print("-" * 80)

for c in cats:
    row = [f"{c:<25}"]
    for name, data in loaded:
        cat_stats = data['categories'].get(c, {})
        passed = cat_stats.get('passed', 0)
        total = cat_stats.get('total', 0)
        acc = (passed / total * 100.0) if total > 0 else 0.0
        row.append(f"{passed:>2}/{total:<2} ({acc:>5.1f}%)    ")
    print(" | ".join(row))

print("-" * 80)
overall_row = [f"{'OVERALL ACCURACY':<25}"]
for name, data in loaded:
    passed = data['total_passed']
    total = data['total_probes']
    acc = data['accuracy_percent']
    overall_row.append(f"{passed:>2}/{total:<2} ({acc:>5.1f}%)    ")
print(" | ".join(overall_row))
