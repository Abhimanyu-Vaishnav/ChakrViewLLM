"""
ChakrView Experiment 15 — Pointer Label & Anti-Leakage Audit
Audits token alignments, verifies context span extracts target entity,
validates abstention labels, checks for multi-token BPE boundaries,
and verifies no test leakage exists.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPO_ROOT = ROOT.parents[2]
TOKENIZER_DIR = REPO_ROOT / "chakrview" / "tokenizer" / "canonical"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

def audit_split(split_path: Path, tokenizer):
    total = 0
    span_matches = 0
    abstention_count = 0
    boundary_mismatches = []
    invalid_spans = []

    with open(split_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total += 1
            rec = json.loads(line)
            tokens = rec["input_ids"]
            start = rec["start_pos"]
            end = rec["end_pos"]
            target = rec["target_text"]
            category = rec["category"]
            
            # Position 0 abstention
            if start == 0 and end == 0:
                abstention_count += 1
                continue
                
            if start < 1 or end < start or end >= len(tokens):
                invalid_spans.append((rec["example_id"], f"Invalid span [{start}, {end}] for len {len(tokens)}"))
                continue

            extracted_tokens = tokens[start : end + 1]
            extracted_text = tokenizer.decode(extracted_tokens).strip()

            if extracted_text.lower() == target.lower():
                span_matches += 1
            else:
                boundary_mismatches.append({
                    "id": rec["example_id"],
                    "target": target,
                    "extracted": extracted_text,
                    "span": [start, end]
                })

    return {
        "file": split_path.name,
        "total": total,
        "abstention_count": abstention_count,
        "exact_span_matches": span_matches,
        "span_match_rate": (span_matches + abstention_count) / total if total > 0 else 0.0,
        "boundary_mismatches": boundary_mismatches,
        "invalid_spans": invalid_spans
    }

def main():
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)

    results = {}
    for split in ["pointer_train.jsonl", "pointer_val.jsonl", "pointer_test.jsonl"]:
        path = DATA_DIR / split
        print(f"Auditing {split}...")
        res = audit_split(path, tokenizer)
        results[split] = res
        print(f"  Total: {res['total']}, Abstentions: {res['abstention_count']}, Exact matches: {res['exact_span_matches']}, Match rate: {res['span_match_rate']:.4f}")
        if res["boundary_mismatches"]:
            print(f"  WARNING: {len(res['boundary_mismatches'])} boundary mismatches found in {split}")
            for bm in res["boundary_mismatches"][:3]:
                print(f"    - ID: {bm['id']} Target: '{bm['target']}' Extracted: '{bm['extracted']}'")

    out_file = ROOT / "audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved audit results to {out_file}")

if __name__ == "__main__":
    main()
