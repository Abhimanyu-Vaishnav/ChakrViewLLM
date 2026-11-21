"""
Summarize attention ablation results
"""
import json

data = json.load(open('experiments/ucip27/attention_ablation/attention_ablation_results.json'))
for ckpt, cases in data.items():
    print(f'=== {ckpt} Attention Ablation ===')
    for c in cases[:2]:
        print(f"Case: {c['prompt_id']} | Target: {c['target']} | Base Top-1: {repr(c['base_top1_token'])} (Rank {c['base_target_rank']})")
        print(f"  All Attention Ablated Top-1: {repr(c['all_attention_ablated_top1'])}, Cos Sim to Base: {c['all_attention_ablated_cos_sim']:.4f}")
        for l in c['layer_ablations']:
            print(f"    Ablate L{l['ablated_layer']}: Top-1={repr(l['top1_token'])}, Tgt Rank={l['target_token_rank']} (Delta: {l['target_rank_delta']}), JS Div={l['js_divergence_from_unablated']:.4f}, Gen={repr(l['generated_text'])}")
