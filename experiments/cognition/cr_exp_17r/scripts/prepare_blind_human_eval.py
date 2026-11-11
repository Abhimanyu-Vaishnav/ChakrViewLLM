import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
RAW_PATH = REPO_ROOT / "experiments/cognition/cr_exp_17r/results/raw_generations.jsonl"
OUT_MD = REPO_ROOT / "experiments/cognition/cr_exp_17r/human_eval/blinded_evaluation_sheet.md"
OUT_JSON = REPO_ROOT / "experiments/cognition/cr_exp_17r/human_eval/human_evaluation_results.json"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

pairs = []
with open(RAW_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))

# Deterministic pseudo-random A/B assignment based on probe number (even = Parent is A, odd = Candidate is A)
blinded_entries = []
for idx, p in enumerate(pairs, 1):
    pid = p["probe_id"]
    cat = p["category"]
    prompt = p["prompt"]
    
    # Randomize order
    if idx % 2 == 0:
        sys_a = p["parent_response"]
        sys_b = p["candidate_response"]
        map_a = "parent"
        map_b = "candidate"
    else:
        sys_a = p["candidate_response"]
        sys_b = p["parent_response"]
        map_a = "candidate"
        map_b = "parent"
        
    blinded_entries.append({
        "probe_id": pid,
        "category": cat,
        "prompt": prompt,
        "system_a": sys_a,
        "system_b": sys_b,
        "mapping": {"system_a": map_a, "system_b": map_b},
        "required_info": p["required_info"]
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "status": "PENDING_HUMAN_EVALUATOR",
        "description": "Blinded evaluation sheet for 100 paired conversational probes. In accordance with Section 11 and Section 16 of the mission specification, artificial model ratings are not fabricated, and coding agent judgment is not substituted for real human evaluations.",
        "completed_human_sample_size": 0,
        "total_probes": len(blinded_entries),
        "blinded_probes": blinded_entries
    }, f, indent=2)

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# CR-EXP-17R: Blinded Human Evaluation Sheet (100 Probes)\n\n")
    f.write("**Status:** PENDING (Awaiting independent human rater scoring)\n\n")
    f.write("## Evaluation Instructions & Rubric (1-5 Scale)\n")
    f.write("Evaluate each response on:\n")
    f.write("1. **Relevance (1-5):** Does the response answer the user or address their conversational topic?\n")
    f.write("2. **Naturalness (1-5):** Is the language grammatically fluent, conversational, and free of loop artifacts?\n")
    f.write("3. **Context Consistency (1-5):** Does the response respect prior turns and refrain from hallucinating conflicting facts?\n")
    f.write("4. **Preference:** `[ ] System A   [ ] System B   [ ] Tie   [ ] Both Poor`\n\n")
    f.write("---\n\n")
    
    for item in blinded_entries:
        f.write(f"### Probe {item['probe_id']} [{item['category']}]\n")
        f.write("```\n" + item["prompt"].strip() + "\n```\n\n")
        f.write(f"- **System A Response:** `{item['system_a']}`\n")
        f.write(f"- **System B Response:** `{item['system_b']}`\n\n")
        f.write("**Rater Scorecard:**\n")
        f.write("- System A: Relevance [ ]  Naturalness [ ]  Consistency [ ]\n")
        f.write("- System B: Relevance [ ]  Naturalness [ ]  Consistency [ ]\n")
        f.write("- Preference: [ ] System A  [ ] System B  [ ] Tie  [ ] Both Poor\n")
        f.write("- Reason: ___________________________________________________\n\n")
        f.write("---\n\n")

print(f"Created {OUT_MD} and {OUT_JSON} successfully.")
