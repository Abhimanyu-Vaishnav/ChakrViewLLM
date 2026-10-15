import json
import re
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096")
ckpt = torch.load(REPO_ROOT / "experiments" / "batch541_580" / "checkpoints" / "candidate_step250_intent_understanding.pt", map_location="cpu", weights_only=False)
model = ChakrMicro(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open(REPO_ROOT / "experiments" / "batch541_580" / "data" / "test_intent_heldout.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

total = len(items)
valid_json_count = 0
intent_matches = 0
arg_matches = 0
clean_eos_count = 0
per_intent_stats = {}
details = []

for item in items:
    exp_obj = json.loads(item["target"])
    exp_intent = exp_obj.get("intent")
    exp_args = exp_obj.get("args", {})

    if exp_intent not in per_intent_stats:
        per_intent_stats[exp_intent] = {
            "total": 0,
            "intent_correct": 0,
            "valid_json": 0,
            "args_correct": 0,
        }
    per_intent_stats[exp_intent]["total"] += 1

    prompt = item["prompt"]
    p_toks = [0] + tokenizer.encode(prompt)
    gen = list(p_toks)
    eos_found = False

    with torch.no_grad():
        for _ in range(80):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            nxt = int(torch.argmax(model(x)[0, -1, :]).item())
            if nxt == 1:
                eos_found = True
                break
            gen.append(nxt)

    if eos_found:
        clean_eos_count += 1

    raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()

    is_valid_json = False
    pred_intent = None
    pred_args = {}

    # Robust JSON parsing
    first_brace = raw_gen.find("{")
    last_brace = raw_gen.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_slice = raw_gen[first_brace:last_brace+1]
        try:
            pred_obj = json.loads(json_slice)
            is_valid_json = True
            pred_intent = pred_obj.get("intent")
            pred_args = pred_obj.get("args", {})
        except Exception:
            is_valid_json = False

    if is_valid_json:
        valid_json_count += 1
        per_intent_stats[exp_intent]["valid_json"] += 1

    intent_correct = (pred_intent == exp_intent)
    if intent_correct:
        intent_matches += 1
        per_intent_stats[exp_intent]["intent_correct"] += 1

    args_correct = False
    if is_valid_json and intent_correct:
        if exp_intent == "calculator":
            exp_expr = str(exp_args.get("expression", "")).replace(" ", "")
            pred_expr = str(pred_args.get("expression", "")).replace(" ", "")
            args_correct = (exp_expr == pred_expr and exp_expr != "")
        elif exp_intent == "text_utility":
            args_correct = (
                pred_args.get("operation") == exp_args.get("operation") and
                pred_args.get("text") == exp_args.get("text")
            )
        elif exp_intent == "concept_definition":
            args_correct = (str(pred_args.get("concept", "")).lower() == str(exp_args.get("concept", "")).lower())
        elif exp_intent == "clarification_needed":
            args_correct = (pred_args.get("missing_field") == exp_args.get("missing_field"))
        elif exp_intent in ("direct_chat", "unsupported", "knowledge_lookup"):
            args_correct = True

    if args_correct:
        arg_matches += 1
        per_intent_stats[exp_intent]["args_correct"] += 1

    details.append({
        "id": item.get("id"),
        "prompt": prompt,
        "expected_intent": exp_intent,
        "predicted_intent": pred_intent,
        "is_valid_json": is_valid_json,
        "intent_correct": intent_correct,
        "args_correct": args_correct,
        "generated": raw_gen,
    })

print(f"Total: {total}")
print(f"Valid JSON Rate: {valid_json_count/total*100:.1f}% ({valid_json_count}/{total})")
print(f"Intent Accuracy: {intent_matches/total*100:.1f}% ({intent_matches}/{total})")
print(f"Argument Accuracy: {arg_matches/total*100:.1f}% ({arg_matches}/{total})")
print(f"Clean EOS Rate: {clean_eos_count/total*100:.1f}% ({clean_eos_count}/{total})")

print("\n--- Per-Intent Breakdown ---")
for cat, stats in per_intent_stats.items():
    tot = stats["total"]
    print(f"  {cat:22s} | N={tot:2d} | Intent Acc: {stats['intent_correct']/tot*100:5.1f}% | Valid JSON: {stats['valid_json']/tot*100:5.1f}% | Args Acc: {stats['args_correct']/tot*100:5.1f}%")
