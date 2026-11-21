"""
Summarize layer tracing results
"""
import json

data = json.load(open('experiments/ucip27/layer_tracing/layer_tracing_results.json'))
for ckpt, cases in data.items():
    print(f'=== {ckpt} ===')
    for c in cases:
        print(f"Case: {c['case_id']} | Target: {c['target_val']} | Top-1: {repr(c['top1_token_str'])}")
        print(f"  Raw Cos Sim={c['raw_logit_cosine_sim']:.4f}, Centered Cos Sim={c['centered_logit_cosine_sim']:.4f}, JS Div={c['jensen_shannon_divergence']:.4f}")
        print(f"  Target Rank={c['target_rank']}, Target Prob={c['target_prob']:.4f}, Margin={c['target_margin_over_top1']:.4f}")
        print("  Layer-wise Attention vs MLP Cosine Similarities (Base vs Value-Replaced Context):")
        for l in c['layer_comparisons']:
            print(f"    Layer {l['layer']}: Attn Cos={l['attn_out_cos_sim']:.4f}, MLP Cos={l['mlp_out_cos_sim']:.4f}, Attn Norm={l['attn_norm_base']:.3f}, MLP Norm={l['mlp_norm_base']:.3f}")
