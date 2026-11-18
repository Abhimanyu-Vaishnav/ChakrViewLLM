import json

models = {
    'uc09': 'experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json',
    'uc09r': 'experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json',
    'uc09s': 'experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json'
}

for name, path in models.items():
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    print(f"=== {name} Evaluator False Negative Scan ===")
    potentially_valid = []
    for e in d['evaluations']:
        if not e['passed']:
            out = e['raw_output'].strip().lower()
            # Check if output is reasonable or contains expected elements
            # e.g., for logic, math, copying, uncertainty, etc.
            if e['domain'] == 'logic' and ('yes' in out or 'no' in out):
                potentially_valid.append((e['id'], e['domain'], e['prompt'], e['raw_output'], e['reason']))
            elif e['domain'] == 'mathematics' and any(c.isdigit() for c in out):
                potentially_valid.append((e['id'], e['domain'], e['prompt'], e['raw_output'], e['reason']))
            elif e['domain'] == 'copying':
                potentially_valid.append((e['id'], e['domain'], e['prompt'], e['raw_output'], e['reason']))
            elif e['domain'] == 'uncertainty':
                potentially_valid.append((e['id'], e['domain'], e['prompt'], e['raw_output'], e['reason']))
    
    print(f"Found {len(potentially_valid)} candidates with non-empty relevant signals.")
    for cid, dom, prompt, out, reason in potentially_valid[:8]:
        print(f"  [{dom}] {cid}: Prompt: {prompt[:40]}... | Out: {out} | Reason: {reason}")
    print("-" * 60)
