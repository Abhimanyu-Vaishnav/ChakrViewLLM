"""
Summarize FFN ablation results
"""
import json

data = json.load(open('experiments/ucip27/ffn_ablation/ffn_ablation_results.json'))
for ckpt, cases in data.items():
    print(f'=== {ckpt} FFN Ablation ===')
    for c in cases[:2]:
        print(f"Case: {c['prompt_id']} | Target: {c['target']} | Base Top-1: {repr(c['base_top1_token'])} (Rank {c['base_target_rank']})")
        print(f"  All FFN Ablated (Attn Only) Top-1: {repr(c['all_ffn_ablated_top1'])}, Target Rank: {c['all_ffn_ablated_target_rank']}, Cos Sim: {c['all_ffn_ablated_cos_sim']:.4f}, JS Div: {c['all_ffn_ablated_js_div']:.4f}")
        for l in c['layer_ablations']:
            print(f"    Ablate FFN L{l['ablated_layer']}: Top-1={repr(l['top1_token'])}, Tgt Rank={l['target_token_rank']} (Delta: {l['target_rank_delta']}), JS Div={l['js_divergence_from_unablated']:.4f}, Gen={repr(l['generated_text'])}")
