"""
Summarize teacher-forcing results script
"""
import json
import numpy as np

data = json.load(open('experiments/ucip26/teacher_forcing/teacher_forcing_comparison.json'))
for m, recs in data.items():
    nlls = [r['teacher_forced_nll'] for r in recs]
    ppls = [r['perplexity'] for r in recs]
    first_p = [r['first_token_prob'] for r in recs]
    print(f'=== {m} Teacher-Forcing ===')
    print(f'NLL: Mean={np.mean(nlls):.3f}, Median={np.median(nlls):.3f}, Min={min(nlls):.3f}, Max={max(nlls):.3f}')
    print(f'Perplexity: Mean={np.mean(ppls):.2f}, Median={np.median(ppls):.2f}')
    print(f'First Target Token Prob: Mean={np.mean(first_p):.4f}, Median={np.median(first_p):.4f}')
    print('Sample Forced Prefix Continuations:')
    for r in recs[:4]:
        print(f"  Target: '{r['target']}' | Prefix: '{r['forced_prefix_token']}' -> Continuation: '{r['forced_prefix_continuation']}'")
