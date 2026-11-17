import json
from collections import defaultdict

with open("D:/Project/ChakrView/experiments/cognition/ucip_07/eval_ucip07_medium.json", "r", encoding="utf-8") as f:
    d = json.load(f)

by_cat = defaultdict(list)
for p in d["evaluations"]:
    by_cat[p["category"]].append(p)

for cat, items in by_cat.items():
    passed = [p for p in items if p["passed"]]
    print(f"\n==================== CATEGORY: {cat} ({len(passed)}/{len(items)}) ====================")
    for p in passed[:5]:
        print(f"[{p['id']}] Q: {p['prompt']}")
        print(f"  Exp KW: {p.get('expected_keywords')}")
        print(f"  Out:    \"{p['generated']}\"")
        print(f"  Check:  {p['details']}")
