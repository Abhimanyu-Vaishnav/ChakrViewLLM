import json
from pathlib import Path

models = {
    'uc09': 'experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json',
    'uc09r': 'experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json',
    'uc09s': 'experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json'
}

for name, path in models.items():
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    evals = data['evaluations']
    passes = [e for e in evals if e['passed']]
    print(f"=== {name} ===")
    print(f"Total: {len(evals)}, Passes: {len(passes)}")
    for p in passes:
        print(f"  {p['id']} [{p['domain']}]: prompt='{p['prompt'][:50]}' out='{p['raw_output']}' reason='{p['reason']}'")
