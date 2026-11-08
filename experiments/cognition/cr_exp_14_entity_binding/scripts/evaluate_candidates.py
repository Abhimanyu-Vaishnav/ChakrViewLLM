"""
CR-EXP-14 Phase 6: Evaluate Trained Candidates on Sealed Benchmark
Evaluates:
1. Candidate A (Best Val Checkpoint)
2. Candidate B (Best Val Checkpoint)
3. Candidate C (Best Val Checkpoint)

Against the sealed 1,200-probe benchmark (benchmark_1200_sealed.json).
Outputs comprehensive evaluation metrics and detailed predictions.
"""

import sys
import json
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BENCHMARK_1200 = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "benchmark_data" / "benchmark_1200_sealed.json"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_eval_model(ckpt_path):
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    m.eval()
    return m

def generate_greedy(model, tok, prompt, max_new_tokens=25):
    enc = tok.encode(prompt)
    curr = list(enc)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([curr], dtype=torch.long)
            logits = model(x)[0, -1]
            nxt = logits.argmax().item()
            curr.append(nxt)
            if nxt == 13 and len(curr) - len(enc) > 3:
                break
    gen_ids = curr[len(enc):]
    try:
        return tok.decode(gen_ids, errors="replace").strip()
    except Exception:
        return "".join([tok.decode([tid], errors="replace") for tid in gen_ids]).strip()

def evaluate_on_sealed_benchmark(model, tok, probes):
    abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state", "unmentioned", "not mention"]
    cat_stats = {}
    tot_strict_hits = 0
    tot_subject_hits = 0
    tot_value_hits = 0
    tot_recency_hits = 0
    details = []

    for p in probes:
        pid = p["id"]
        cat = p["category"]
        prompt = p["prompt"]
        subj = p.get("subject", "")
        tgt = p.get("expected_target", "")
        stale = p.get("earlier_value", "")
        distractors = p.get("distractor_values", [])

        gen = generate_greedy(model, tok, prompt)
        is_abstained = any(cue in gen.lower() for cue in abstain_cues)

        has_subj = subj.lower() in gen.lower() if subj else False
        has_tgt = tgt.lower() in gen.lower() if tgt else False
        has_stale = stale.lower() in gen.lower() if stale else False
        has_distractor = any(d.lower() in gen.lower() for d in distractors)

        is_strict_hit = False
        is_value_hit = False
        is_recency_hit = False

        if cat == "missing_information_abstention":
            is_strict_hit = is_abstained
            is_value_hit = is_abstained
            has_subj = True
        elif cat == "counterfactual_reassignment":
            is_value_hit = has_tgt
            is_recency_hit = has_tgt and not has_stale
            is_strict_hit = is_recency_hit
        else:
            is_value_hit = has_tgt
            is_recency_hit = has_tgt and not has_stale
            is_strict_hit = has_subj and has_tgt and not has_stale and not has_distractor

        if is_strict_hit: tot_strict_hits += 1
        if has_subj: tot_subject_hits += 1
        if is_value_hit: tot_value_hits += 1
        if is_recency_hit: tot_recency_hits += 1

        cat_stats.setdefault(cat, {
            "strict_hits": 0, "subject_hits": 0, "value_hits": 0, "recency_hits": 0, "total": 0
        })
        cat_stats[cat]["total"] += 1
        if is_strict_hit: cat_stats[cat]["strict_hits"] += 1
        if has_subj: cat_stats[cat]["subject_hits"] += 1
        if is_value_hit: cat_stats[cat]["value_hits"] += 1
        if is_recency_hit: cat_stats[cat]["recency_hits"] += 1

        details.append({
            "id": pid,
            "category": cat,
            "subject": subj,
            "expected_target": tgt,
            "earlier_value": stale,
            "generated": gen,
            "has_subj": has_subj,
            "has_tgt": has_tgt,
            "has_stale": has_stale,
            "is_strict_hit": is_strict_hit
        })

    summary = {}
    tot_probes = len(probes)
    for cat, d in sorted(cat_stats.items()):
        summary[cat] = {
            "total": d["total"],
            "strict_hits": d["strict_hits"],
            "strict_accuracy_pct": round(d["strict_hits"] / d["total"] * 100, 2),
            "subject_accuracy_pct": round(d["subject_hits"] / d["total"] * 100, 2),
            "value_accuracy_pct": round(d["value_hits"] / d["total"] * 100, 2),
            "recency_accuracy_pct": round(d["recency_hits"] / d["total"] * 100, 2)
        }
    summary["overall"] = {
        "total_probes": tot_probes,
        "strict_hits": tot_strict_hits,
        "strict_accuracy_pct": round(tot_strict_hits / tot_probes * 100, 2),
        "subject_accuracy_pct": round(tot_subject_hits / tot_probes * 100, 2),
        "value_accuracy_pct": round(tot_value_hits / tot_probes * 100, 2),
        "recency_accuracy_pct": round(tot_recency_hits / tot_probes * 100, 2)
    }
    return summary, details

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(BENCHMARK_1200, "r", encoding="utf-8") as f:
        probes_1200 = json.load(f)

    candidates = {
        "candidate_a": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_14_entity_binding/checkpoints/candidate_a/cr_exp_14_candidate_a_best_val.pt",
            "hash": "58f51bc249aec77235bbdd07179dc130f72e24947500b05fa092f10752a8f033"
        },
        "candidate_b": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_14_entity_binding/checkpoints/candidate_b/cr_exp_14_candidate_b_best_val.pt",
            "hash": "e838eb33e6e49cef68e6afb12e7f93bd788f1259940647dce7ef70717f512cf2"
        },
        "candidate_c": {
            "path": REPO_ROOT / "experiments/cognition/cr_exp_14_entity_binding/checkpoints/candidate_c/cr_exp_14_candidate_c_best_val.pt",
            "hash": "32f1066f4a6b0c652d42991bcc8c1427b63854a55483ffc8a32b726512600d12"
        }
    }

    results = {}
    for name, info in candidates.items():
        print(f"\n[Candidate Evaluation] Evaluating {name} on 1,200 Sealed Probes...")
        t0 = time.time()
        m = load_eval_model(info["path"])
        summary, details = evaluate_on_sealed_benchmark(m, tok, probes_1200)
        dt = time.time() - t0
        print(f"[{name}] Done in {dt:.1f}s. Strict Acc: {summary['overall']['strict_accuracy_pct']}% ({summary['overall']['strict_hits']}/1200)")
        print(f"      Subject Acc: {summary['overall']['subject_accuracy_pct']}%, Value Acc: {summary['overall']['value_accuracy_pct']}%")

        results[name] = {
            "checkpoint_hash": info["hash"],
            "summary": summary
        }

        det_file = OUT_DIR / f"{name}_sealed_detailed.json"
        with open(det_file, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)

    out_path = OUT_DIR / "candidate_evaluations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll candidate evaluations saved to {out_path}")

if __name__ == "__main__":
    main()
