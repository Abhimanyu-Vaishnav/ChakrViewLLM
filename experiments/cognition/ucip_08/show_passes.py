import json

for name, p in [
    ('UCIP-06 Medium', 'experiments/cognition/ucip_08/eval_ucip06_medium_owner_reality.json'),
    ('UCIP-07 Medium', 'experiments/cognition/ucip_08/eval_ucip07_medium_owner_reality.json'),
    ('UCIP-08 Candidate', 'experiments/cognition/ucip_08/eval_ucip08_medium_owner_reality.json')
]:
    d = json.load(open(p, encoding='utf-8'))
    print(f"=== {name} Passes ({d['total_passed']}/{d['total_probes']}) ===")
    for item in d['evaluations']:
        if item['passed']:
            print(f"  [{item['category']}] Q: {item['prompt']}")
            print(f"    -> A: {repr(item['generated'])}")
    print()
