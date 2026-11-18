import json

with open('experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json', encoding='utf-8') as f:
    d = json.load(f)
logic_evals = [e for e in d['evaluations'] if e['domain'] == 'logic']
for e in logic_evals[:20]:
    print(f"{e['id']}: out='{e['raw_output']}' passed={e['passed']} reason='{e['reason']}'")
