"""
UCIP-26 Phase 7: Training Distribution Comparison.
Inspects:
- Frequency of top attractor tokens ('Analyst', '#', '[', 'A') in:
  1. UCIP-20 training targets (bind_train_1600.jsonl)
  2. Stage C corpus texts
- Distribution of first response tokens in BIND training
- Target length distribution in BIND training vs UCIP-23 benchmark
- Presence of delimiter '#' and bracketed date loops in training sets.
"""

from collections import Counter
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_training_distribution_comparison():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    # 1. Inspect UCIP-20 BIND training data
    bind_train_path = "data/sft/ucip20_bind/bind_train_1600.jsonl"
    first_tokens = Counter()
    responses = []
    response_lengths = []

    with open(bind_train_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            resp = rec.get("response", "").strip()
            responses.append(resp)
            toks = tok.encode(resp)
            if toks:
                first_tokens[toks[0]] += 1
                response_lengths.append(len(toks))

    top_first_toks = [
        {"token_id": t, "token_str": tok.decode([t]), "count": cnt, "pct": cnt / len(responses) * 100}
        for t, cnt in first_tokens.most_common(10)
    ]

    # Check frequency of 'Analyst', 'A', '[', '#' in BIND responses
    attractor_check = {
        "Analyst_in_responses": sum(1 for r in responses if "Analyst" in r),
        "bracket_in_responses": sum(1 for r in responses if "[" in r),
        "hash_in_responses": sum(1 for r in responses if "#" in r),
        "abstain_in_responses": sum(1 for r in responses if "Information not provided in context" in r)
    }

    # 2. Check UCIP-23 benchmark response lengths
    bench_lengths = []
    bench_first_tokens = Counter()
    for p in sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tgt = rec.get("target_response", "").strip()
                t_toks = tok.encode(tgt)
                if t_toks:
                    bench_first_tokens[t_toks[0]] += 1
                    bench_lengths.append(len(t_toks))

    top_bench_first_toks = [
        {"token_id": t, "token_str": tok.decode([t]), "count": cnt, "pct": cnt / len(bench_lengths) * 100}
        for t, cnt in bench_first_tokens.most_common(10)
    ]

    results = {
        "bind_training_records": len(responses),
        "bind_target_lengths": {
            "min": min(response_lengths),
            "max": max(response_lengths),
            "mean": sum(response_lengths) / len(response_lengths)
        },
        "bind_top_first_tokens": top_first_toks,
        "bind_attractor_frequencies": attractor_check,
        "benchmark_records": len(bench_lengths),
        "benchmark_target_lengths": {
            "min": min(bench_lengths),
            "max": max(bench_lengths),
            "mean": sum(bench_lengths) / len(bench_lengths)
        },
        "benchmark_top_first_tokens": top_bench_first_toks
    }

    out_file = "experiments/ucip26/training_distribution/training_distribution_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Phase 7 Training Distribution Comparison Complete.")
    print("Top first tokens in UCIP-20 BIND training:")
    for t in top_first_toks[:5]:
        print(f"  Token {t['token_id']} ('{t['token_str']}'): {t['count']} ({t['pct']:.1f}%)")
    print("\nBIND Attractor Frequencies:")
    for k, v in attractor_check.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_training_distribution_comparison()
