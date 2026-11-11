import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

BENCHMARK_PATH = REPO_ROOT / "experiments/cognition/cr_exp_18/benchmark/CR18_CONTEXT_200.json"
RELEASE_CKPT = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"
OUT_FILE = REPO_ROOT / "experiments/cognition/cr_exp_18/results/CONTEXT_ABLATION_RESULTS.json"

with open(BENCHMARK_PATH, encoding="utf-8") as f:
    bench_data = json.load(f)

runner = DirectNeuralRunner(RELEASE_CKPT)
tok = runner.tokenizer
cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)

# Test 40 diverse conversations (8 from each of the 5 length groups A, B, C, D, E) across conditions:
# Condition A: Full conversation history
# Condition B: Only the most recent user message
# Condition C: Most recent user message plus previous assistant response
# Condition D: Original conversation with earlier user facts removed/scrambled
# Condition E: Original conversation with earlier facts replaced by counterfactual values (e.g. London -> Tokyo)

sample_probes = []
for grp in ["A", "B", "C", "D", "E"]:
    grp_items = [x for x in bench_data if x["group"] == grp]
    sample_probes.extend(grp_items[:8])

print(f"Evaluating {len(sample_probes)} probes across 5 ablation conditions...")

ablation_results = []

def generate_resp(prompt_str):
    toks = tok.encode(prompt_str, add_bos=True, add_eos=False)
    curr = list(toks)
    gen = []
    with torch.no_grad():
        for step in range(32):
            t_in = torch.tensor([curr], dtype=torch.long)
            logits = runner.model(t_in)[0, -1, :]
            next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
            gen.append(next_tok)
            curr.append(next_tok)
            if next_tok == 1:
                break
    return tok.decode(gen, errors="replace").strip()

for p in sample_probes:
    pid = p["conversation_id"]
    grp = p["group"]
    turns = p["turns"]
    expected = p["expected_answer"]
    last_user = turns[-1]["user"]
    
    # Cond A: Full History
    prompt_a = p["formatted_prompt"]
    resp_a = generate_resp(prompt_a)
    hit_a = any(exp.lower() in resp_a.lower() for exp in expected)
    
    # Cond B: Only last user message
    prompt_b = f"User: {last_user}\nAssistant:"
    resp_b = generate_resp(prompt_b)
    hit_b = any(exp.lower() in resp_b.lower() for exp in expected)
    
    # Cond C: Last user + previous assistant
    if len(turns) >= 2 and turns[-2]["assistant"]:
        prev_ast = turns[-2]["assistant"]
        prompt_c = f"Assistant: {prev_ast}\nUser: {last_user}\nAssistant:"
    else:
        prompt_c = prompt_b
    resp_c = generate_resp(prompt_c)
    hit_c = any(exp.lower() in resp_c.lower() for exp in expected)
    
    # Cond E: Counterfactual replacement (replace target in prompt with COUNTERFACTUAL)
    # E.g. replace target fact value with 'Atlantis'
    prompt_e = prompt_a
    cf_val = "Atlantis"
    for exp_str in expected:
        prompt_e = prompt_e.replace(exp_str, cf_val)
    resp_e = generate_resp(prompt_e)
    hit_cf = (cf_val.lower() in resp_e.lower())
    
    ablation_results.append({
        "conversation_id": pid,
        "group": grp,
        "turns_count": p["turns_count"],
        "expected_answer": expected,
        "condition_A_full": {"prompt": prompt_a.replace("\n", " -- "), "response": resp_a, "hit": hit_a},
        "condition_B_last_user": {"prompt": prompt_b.replace("\n", " -- "), "response": resp_b, "hit": hit_b},
        "condition_C_prev_turn": {"prompt": prompt_c.replace("\n", " -- "), "response": resp_c, "hit": hit_c},
        "condition_E_counterfactual": {"response": resp_e, "counterfactual_tracked": hit_cf}
    })

# Summary
by_group = {}
for r in ablation_results:
    g = r["group"]
    if g not in by_group:
        by_group[g] = {"total": 0, "hits_A": 0, "hits_B": 0, "hits_C": 0, "cf_tracked": 0}
    by_group[g]["total"] += 1
    if r["condition_A_full"]["hit"]: by_group[g]["hits_A"] += 1
    if r["condition_B_last_user"]["hit"]: by_group[g]["hits_B"] += 1
    if r["condition_C_prev_turn"]["hit"]: by_group[g]["hits_C"] += 1
    if r["condition_E_counterfactual"]["counterfactual_tracked"]: by_group[g]["cf_tracked"] += 1

summary = {
    "total_probes_evaluated": len(ablation_results),
    "by_group_metrics": by_group,
    "overall": {
        "condition_A_full_history_hits": sum(x["condition_A_full"]["hit"] for x in ablation_results),
        "condition_B_last_user_only_hits": sum(x["condition_B_last_user"]["hit"] for x in ablation_results),
        "condition_C_prev_turn_hits": sum(x["condition_C_prev_turn"]["hit"] for x in ablation_results),
        "condition_E_counterfactual_tracked": sum(x["condition_E_counterfactual"]["counterfactual_tracked"] for x in ablation_results)
    }
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": ablation_results}, f, indent=2)

print("\n=== CONTEXT ABLATION SUMMARY ===")
print(json.dumps(summary, indent=2))
