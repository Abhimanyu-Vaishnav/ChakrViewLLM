import json

with open('experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json') as f:
    bench = json.load(f)
with open('experiments/conversation/cr_exp_17/v091_eval_results.json') as f:
    v091 = json.load(f)
with open('experiments/conversation/cr_exp_17/baseline_eval_results.json') as f:
    base = json.load(f)
parent = base['Conversational SFT Parent (Step 200)']

# Select 100 diverse probes (first 8-9 of each of the 12 categories)
selected = []
for p, p_par in zip(v091['records'], parent['records']):
    cat = p['category']
    count_for_cat = sum(1 for s in selected if s['category'] == cat)
    if count_for_cat < 9 and len(selected) < 100:
        selected.append({
            'id': p['id'],
            'category': p['category'],
            'prompt': p['prompt'],
            'baseline_parent_response': p_par['response'],
            'v091_response': p['response']
        })

print(f"Selected {len(selected)} probes for blind human evaluation.")
with open('experiments/conversation/cr_exp_17/blind_human_eval_100.json', 'w') as f:
    json.dump(selected, f, indent=2)

with open('experiments/conversation/cr_exp_17/blind_human_eval_100.md', 'w', encoding='utf-8') as f:
    f.write("# CHAKRVIEW v0.9.1 — BLIND HUMAN EVALUATION SHEET (100 PROBES)\n\n")
    f.write("## Evaluation Rubric (1 to 5 Scale)\n")
    f.write("- **1: Degenerate / Broken:** Incoherent word salad, broken tokens, or pure repetition loop.\n")
    f.write("- **2: Refusal Sink / Irrelevant:** Non-responsive repetition of refusal, ignores user prompt entirely.\n")
    f.write("- **3: Weak / Generic:** Grammatically formed but vague, partially hallucinated, or canned greeting response.\n")
    f.write("- **4: Good:** Coherent, semantically relevant, appropriate conversational tone and language.\n")
    f.write("- **5: Excellent:** Precise, highly conversational, maintains multi-turn context and factual clarity.\n\n")
    f.write("---\n\n")
    for item in selected:
        pid = item['id']
        pcat = item['category']
        prompt_txt = item['prompt'].strip()
        resp_a = item['baseline_parent_response']
        resp_b = item['v091_response']
        f.write(f"### Probe {pid} [{pcat}]\n")
        f.write("```\n" + prompt_txt + "\n```\n\n")
        f.write(f"- **System A Response:** `{resp_a}`\n")
        f.write(f"- **System B Response:** `{resp_b}`\n\n")
        f.write("**Human Evaluation Scoring:**\n")
        f.write("- System A Score (1-5): [ ]\n")
        f.write("- System B Score (1-5): [ ]\n")
        f.write("- Preference: [ ] System A  [ ] System B  [ ] Tie\n\n")
        f.write("---\n\n")

print("Created blind_human_eval_100.json and blind_human_eval_100.md")
