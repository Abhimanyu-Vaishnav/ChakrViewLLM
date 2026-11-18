import sys
from pathlib import Path
REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import collections
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def verify_dataset(path: Path, tok):
    total = 0
    categories = collections.Counter()
    unique_prompts = set()
    unique_targets = set()
    raw_lines = set()
    duplicates = 0
    word_counter = collections.Counter()
    token_counter = collections.Counter()
    numeric_words = collections.Counter()
    template_phrases = collections.Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str in raw_lines:
                duplicates += 1
            raw_lines.add(line_str)
            total += 1
            rec = json.loads(line_str)
            cat = rec.get("category", rec.get("domain", "unknown"))
            categories[cat] += 1
            prompt = rec.get("prompt", "")
            target = rec.get("target", "")
            unique_prompts.add(prompt)
            unique_targets.add(target)

            # Word counting
            words = target.lower().split()
            for w in words:
                word_counter[w] += 1
                # clean numeric check
                w_num = "".join(ch for ch in w if ch.isdigit())
                if w_num:
                    numeric_words[w_num] += 1

            if "remaining." in target.lower():
                template_phrases["remaining."] += 1
            if "in total." in target.lower():
                template_phrases["in total."] += 1

            # Tokenizer token counting
            tgt_ids = tok.encode(target, add_bos=False, add_eos=False)
            for tid in tgt_ids:
                token_counter[tid] += 1

    # Map top token ids back to token strings
    top_tokens = []
    for tid, count in token_counter.most_common(25):
        t_str = tok.decode([tid])
        top_tokens.append({"id": tid, "token_str": repr(t_str), "count": count})

    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "total_records": total,
        "duplicate_records": duplicates,
        "unique_prompts": len(unique_prompts),
        "unique_targets": len(unique_targets),
        "categories": dict(categories.most_common()),
        "top_target_words": word_counter.most_common(20),
        "numeric_word_frequencies": numeric_words.most_common(20),
        "template_phrases": dict(template_phrases),
        "top_target_tokens": top_tokens
    }

def main():
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    datasets = {
        "ucip09_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl",
        "ucip09_val": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_validation.jsonl",
        "ucip09r_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl",
        "ucip09r_val": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_val.jsonl",
        "ucip09s_balanced_train": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl",
        "ucip09s_balanced_val": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_val.jsonl"
    }

    results = {}
    for name, p in datasets.items():
        print(f"Auditing {name}...")
        res = verify_dataset(p, tok)
        results[name] = res

    with open(REPO_ROOT / "experiments" / "ucip09s_r1" / "evaluations" / "dataset_token_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done. Saved to experiments/ucip09s_r1/evaluations/dataset_token_verification.json")

if __name__ == "__main__":
    main()
