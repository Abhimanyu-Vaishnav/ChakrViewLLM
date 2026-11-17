import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
DIAG_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "pipeline_diagnostics_raw.json"

data = json.load(open(DIAG_FILE, encoding="utf-8"))

for ckpt_key in ["ucip07_medium", "ucip08_medium"]:
    print(f"\n==========================================")
    print(f"DECODING FACTORIAL ANALYSIS: {ckpt_key}")
    print(f"==========================================")
    arms = data[ckpt_key]["arms"]
    
    # Calculate statistics across arms
    print(f"{'Arm':<25} | {'Avg Len':<8} | {'EOS %':<8} | {'Sample Probe 0 (Greeting)':<45}")
    print("-" * 90)
    for arm_name, results in arms.items():
        avg_len = sum(r["gen_token_count"] for r in results) / len(results)
        eos_pct = sum(1 for r in results if r["ended_with_eos"]) / len(results) * 100
        sample_text = repr(results[0]["generated_text"][:40])
        print(f"{arm_name:<25} | {avg_len:<8.1f} | {eos_pct:<8.1f} | {sample_text:<45}")

    print("\n--- Token Traces (Step 0 Entropy & Top-1 prob) ---")
    traces = data[ckpt_key].get("token_level_traces", [])
    for t in traces:
        p_name = t["prompt"][:30]
        step0 = t["step_diagnostics"][0] if t["step_diagnostics"] else None
        if step0:
            top1 = step0["top5"][0]
            print(f"  [{t['category']}] '{p_name}': Entropy={step0['entropy']:.2f}, Top1='{top1['text']}' (p={top1['prob']:.3f})")
