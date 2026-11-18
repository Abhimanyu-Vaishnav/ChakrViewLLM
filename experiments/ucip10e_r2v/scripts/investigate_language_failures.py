import json
from collections import Counter

models = {
    'uc09': 'experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json',
    'uc09r': 'experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json',
    'uc09s': 'experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json'
}

for name, path in models.items():
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    lang_evals = [e for e in d['evaluations'] if e['domain'] == 'language']
    reasons = Counter([e['reason'] for e in lang_evals])
    print(f"=== {name} Language Failures ({len(lang_evals)}) ===")
    for r, count in reasons.most_common(10):
        print(f"  {count:2d} : {r}")
    
    print("\nSample Language Outputs (first 5):")
    for e in lang_evals[:5]:
        print(f"  ID: {e['id']}")
        print(f"    Prompt: {e['prompt']}")
        print(f"    Raw: {e['raw_output']}")
        print(f"    Reason: {e['reason']}")
    print("-" * 60)
