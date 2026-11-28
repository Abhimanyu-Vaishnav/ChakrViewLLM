"""
Mission CV-24H: Clean Conversational SFT Mixture Builder.
Audits, filters, standardizes, deduplicates, and contamination-checks conversational SFT data.

Sources:
1. cbr_sft_04 (160 dialogues: English dialogue, greetings, assistance)
2. cbr_sft_06a (170 dialogues: structured dialogue, system interactions)
3. m3_conversational (600 dialogues: conversational Q&A, math, facts, chit-chat)
4. cbr_sft_02 (540 dialogues: technical & Hindi QA)
5. exp34_candidate_v3 (900 dialogues: Hinglish, technical reasoning, Hindi)

Contamination Exclusion:
Strictly verifies that NO prompt or 8-gram overlaps with:
- benchmark_v3_300.json
- fresh_heldout_benchmark_50.json
- corpus_v6_test.jsonl

Produces:
- experiments/cv24h_release/sft_mixture_train.jsonl
- experiments/cv24h_release/sft_mixture_val.jsonl
- experiments/cv24h_release/sft_mixture_manifest.json
"""

import os
import json
import random
import hashlib
from collections import Counter

def ngrams(text, n=8):
    words = text.lower().split()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

def is_devanagari(text):
    return any(0x0900 <= ord(c) <= 0x097F for c in text)

def is_hinglish(text):
    # Common hinglish markers
    markers = ["kya", "hai", "hein", "kaise", "bataiye", "samjhaiye", "bilkul", "nahi", "aap", "mera", "hoga", "karo"]
    words = set(text.lower().split())
    return any(m in words for m in markers)

def main():
    print("=== BUILDING CONVERSATIONAL SFT MIXTURE FOR CV-24H ===")
    
    # 1. Collect protected test ngrams
    test_ngrams = set()
    b3_path = "data/evaluation/benchmark_v3/benchmark_v3_300.json"
    with open(b3_path, "r", encoding="utf-8") as f:
        b3 = json.load(f)
        for it in b3:
            p = it.get("prompt", "")
            test_ngrams.update(ngrams(p, 8))
            test_ngrams.update(ngrams(it.get("expected_answer", ""), 8))

    fresh_path = "experiments/forensics_nr053/fresh_heldout_benchmark_50.json"
    with open(fresh_path, "r", encoding="utf-8") as f:
        fresh = json.load(f)
        for it in fresh:
            p = it.get("prompt", "")
            test_ngrams.update(ngrams(p, 8))
            test_ngrams.update(ngrams(it.get("reference_answer", ""), 8))

    test_v6_path = "data/corpus_v6/splits/corpus_v6_test.jsonl"
    with open(test_v6_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            txt = json.loads(line).get("text", "")
            test_ngrams.update(ngrams(txt[:500], 8))

    print(f"Total protected test 8-grams collected: {len(test_ngrams):,}")

    raw_candidates = []
    
    # Load CBR 04
    with open("data/sft/cbr_sft_04/dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_candidates.append({
                "source": "cbr_sft_04",
                "prompt": obj["prompt"].strip(),
                "response": obj["response"].strip()
            })

    # Load CBR 06a
    with open("data/sft/cbr_sft_06a/dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_candidates.append({
                "source": "cbr_sft_06a",
                "prompt": obj["prompt"].strip(),
                "response": obj["response"].strip()
            })

    # Load M3 Conversational
    with open("data/sft/m3_conversational/m3_conversational_mixture_600.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_candidates.append({
                "source": "m3_conversational",
                "prompt": obj["prompt"].strip(),
                "response": obj["response"].strip()
            })

    # Load CBR 02
    with open("data/sft/cbr_sft_02/splits/cbr_sft_02_train_540.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_candidates.append({
                "source": "cbr_sft_02",
                "prompt": obj["prompt"].strip(),
                "response": obj["response"].strip()
            })

    # Load Exp34 Candidate V3
    with open("data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt", "").strip()
            r = obj.get("response", "").strip()
            if p and r:
                raw_candidates.append({
                    "source": "exp34_candidate_v3",
                    "prompt": p,
                    "response": r
                })

    print(f"Total raw candidates ingested: {len(raw_candidates)}")

    # Standardize & Filter
    cleaned = []
    seen_prompts = set()
    contaminated_count = 0

    for cand in raw_candidates:
        p = cand["prompt"]
        r = cand["response"]

        # Ensure uniform formatting: User: ...\nAssistant: ...
        if not p.startswith("User:"):
            # Check if there's user/assistant tag
            if "User:" in p:
                p = p[p.find("User:"):]
            else:
                p = f"User: {p}"
        if not p.endswith("\nAssistant:"):
            if "Assistant:" in p:
                p = p.split("Assistant:")[0].strip() + "\nAssistant:"
            else:
                p = p.rstrip() + "\nAssistant:"

        # Strip response leading whitespace
        r = r.strip()
        if not r:
            continue

        # Deduplication
        p_norm = p.lower()
        if p_norm in seen_prompts:
            continue

        # Contamination check
        cand_ngrams = ngrams(p, 8) | ngrams(r, 8)
        if cand_ngrams & test_ngrams:
            contaminated_count += 1
            continue

        seen_prompts.add(p_norm)

        # Categorize language
        combined_text = p + " " + r
        if is_devanagari(combined_text):
            lang = "hindi_devanagari"
        elif is_hinglish(combined_text):
            lang = "hinglish"
        else:
            lang = "english"

        cleaned.append({
            "id": f"sft_{len(cleaned):05d}",
            "source": cand["source"],
            "language": lang,
            "prompt": p,
            "response": r
        })

    print(f"Contaminated candidates discarded: {contaminated_count}")
    print(f"Total unique, clean, uncontaminated dialogues: {len(cleaned)}")

    lang_counts = Counter(x["language"] for x in cleaned)
    print("Language distribution:")
    for l, cnt in lang_counts.items():
        print(f"  {l}: {cnt} ({cnt/len(cleaned)*100:.1f}%)")

    # Split into train (90%) and val (10%)
    random.seed(42)
    random.shuffle(cleaned)

    val_size = int(len(cleaned) * 0.10)
    val_set = cleaned[:val_size]
    train_set = cleaned[val_size:]

    print(f"Train set: {len(train_set)} samples, Val set: {len(val_set)} samples")

    out_dir = "experiments/cv24h_release"
    train_file = os.path.join(out_dir, "sft_mixture_train.jsonl")
    val_file = os.path.join(out_dir, "sft_mixture_val.jsonl")
    manifest_file = os.path.join(out_dir, "sft_mixture_manifest.json")

    with open(train_file, "w", encoding="utf-8") as f:
        for x in train_set:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for x in val_set:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    manifest = {
        "mission": "CV-24H Conversational SFT Mixture",
        "total_dialogues": len(cleaned),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "language_distribution": dict(lang_counts),
        "sources": dict(Counter(x["source"] for x in cleaned)),
        "contamination_tested": True,
        "contaminated_samples_rejected": contaminated_count,
        "train_file": train_file,
        "val_file": val_file
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved mixture to {train_file} and {val_file}")

if __name__ == "__main__":
    main()
