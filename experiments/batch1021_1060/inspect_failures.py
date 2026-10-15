import json

with open('experiments/batch1021_1060/reports/FINAL_METRICS.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

failures = [i for i in d['details'] if not i['passed']]
print(f"Total failures: {len(failures)}")
for f in failures:
    print(f"ID {f['id']} [{f['category']}] Exp: {f['expected_type']} Got: {f['decision']}")
    print(f"   Query: {f['query']}")
    print(f"   Answer: {f['answer']}")
    print("-" * 50)
