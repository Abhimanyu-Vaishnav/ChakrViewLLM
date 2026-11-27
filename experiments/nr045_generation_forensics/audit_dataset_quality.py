"""
experiments/nr045_generation_forensics/audit_dataset_quality.py
Phase 3: Dataset Quality & Target Forensics on data/corpus_v1/master_corpus_train.jsonl
Measures:
- Sample count, token length distributions
- Language counts
- Assistant response quality & length
- Empty or truncated responses
- Repetitive / boilerplate patterns
- Prompt leakage / prompt formatting mismatch
- Concrete sample inspection across all language groups
"""

import json
from collections import Counter
from pathlib import Path

def audit_dataset():
    path = "data/corpus_v1/master_corpus_train.jsonl"
    print(f"=== AUDITING DATASET: {path} ===")

    total = 0
    empty_resp = 0
    short_resp = 0
    langs = Counter()
    cats = Counter()
    resp_lengths = []
    prompt_lengths = []
    
    boilerplate_matches = Counter()
    boilerplate_phrases = [
        "Scientific research demonstrates",
        "API programming",
        "Station 364 is managed",
        "A remainder of",
        "लॉकर",
        "गुप्त",
        "उत्तर #",
        "Summary #"
    ]

    samples_by_lang = {"en": [], "hi": [], "hinglish": [], "sa": []}

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            total += 1
            d = json.loads(line)
            l = d.get("language", "en")
            c = d.get("category", "unknown")
            p = d.get("prompt", "").strip()
            r = d.get("response", "").strip()

            langs[l] += 1
            cats[c] += 1

            p_len = len(p.split())
            r_len = len(r.split())
            prompt_lengths.append(p_len)
            resp_lengths.append(r_len)

            if not r: empty_resp += 1
            if r_len < 3: short_resp += 1

            for bp in boilerplate_phrases:
                if bp.lower() in r.lower():
                    boilerplate_matches[bp] += 1

            if len(samples_by_lang.get(l, [])) < 5:
                samples_by_lang[l].append({"id": d.get("id"), "prompt": p, "response": r[:160]})

    print(f"Total Records: {total}")
    print(f"Languages: {dict(langs)}")
    print(f"Empty responses: {empty_resp} | Very short responses (<3 words): {short_resp}")
    print(f"Prompt word length: min={min(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.1f}, max={max(prompt_lengths)}")
    print(f"Response word length: min={min(resp_lengths)}, avg={sum(resp_lengths)/len(resp_lengths):.1f}, max={max(resp_lengths)}")
    
    print("\nBoilerplate / Synthetic Distillation Patterns in Responses:")
    for bp, cnt in boilerplate_matches.most_common():
        print(f"  '{bp}': {cnt} occurrences ({cnt/total*100:.1f}%)")

    out_file = Path("experiments/nr045_generation_forensics/dataset_forensics_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_records": total,
            "languages": dict(langs),
            "categories": dict(cats.most_common(20)),
            "boilerplate_frequencies": dict(boilerplate_matches),
            "samples": samples_by_lang
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDataset forensics summary saved to {out_file}")

if __name__ == "__main__":
    audit_dataset()
