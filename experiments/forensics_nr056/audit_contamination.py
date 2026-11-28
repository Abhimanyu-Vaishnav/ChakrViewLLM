import os
import sys
import json
import re
import unicodedata

sys.path.insert(0, os.path.abspath("."))

def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

with open("data/evaluation/benchmark_v2/benchmark_v2_100.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Collect all training texts across available training corpora
corpora_files = [
    ("Corpus_V1", "data/corpus_v1/master_corpus_train.jsonl"),
    ("Corpus_V2", "data/corpus_v2/corpus_v2_train.jsonl"),
    ("Corpus_V3", "data/corpus_v3/corpus_v3_train.jsonl"),
    ("Corpus_V4", "data/corpus_v4/corpus_v4_train.jsonl"),
    ("NR051_Pretrain", "data/pretraining_nr051/pretrain_train.jsonl")
]

training_records = []
for c_name, c_path in corpora_files:
    if not os.path.exists(c_path): continue
    with open(c_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            txt = d.get("text", "")
            if not txt and "prompt" in d and "response" in d:
                txt = f"{d['prompt']} {d['response']}"
            training_records.append((c_name, normalize(txt)))

print(f"Loaded {len(training_records)} training records across all corpora.")

results = []
summary = {
    "CONFIRMED_EXACT": 0,
    "CONFIRMED_NEAR_DUPLICATE": 0,
    "ANSWER_ONLY_OVERLAP": 0,
    "FACTUAL_EXPOSURE": 0,
    "NO_MATCH_FOUND": 0
}

for item in benchmark:
    p_id = item["id"]
    p_norm = normalize(item["prompt"])
    ans_norm = normalize(item["expected_answer"])
    alts_norm = [normalize(a) for a in item.get("acceptable_alternatives", [])]
    all_answers = [ans_norm] + alts_norm
    
    classification = "NO_MATCH_FOUND"
    evidence = []
    
    # Check exact and near duplicate prompts
    p_tokens = set(p_norm.split())
    
    for c_name, c_text in training_records:
        if p_norm in c_text:
            classification = "CONFIRMED_EXACT"
            evidence.append(f"Exact prompt in {c_name}")
            break
            
        # Check near duplicate (high token jaccard or subset)
        if len(p_tokens) >= 4:
            # Check 4-gram overlap
            p_words = p_norm.split()
            four_grams = [" ".join(p_words[i:i+4]) for i in range(len(p_words)-3)]
            matched_grams = [fg for fg in four_grams if fg in c_text]
            if len(matched_grams) >= max(1, len(four_grams) * 0.75):
                classification = "CONFIRMED_NEAR_DUPLICATE"
                evidence.append(f"Near-duplicate phrasing in {c_name}: {matched_grams[0]}")
                break
                
        # Check factual exposure: does training text link key entity with target answer?
        # E.g. Japan and Tokyo, France and Paris, Australia and Canberra
        for ans in all_answers:
            if len(ans) >= 3 and f" {ans} " in f" {c_text} ":
                # Check if prompt main nouns are in c_text
                key_nouns = [w for w in p_tokens if w not in {"what", "is", "the", "capital", "of", "city", "in", "and", "or", "to", "which", "how", "many", "does"}]
                if any(noun in c_text for noun in key_nouns if len(noun) >= 4):
                    if classification not in ("CONFIRMED_EXACT", "CONFIRMED_NEAR_DUPLICATE"):
                        classification = "FACTUAL_EXPOSURE"
                        evidence.append(f"Fact link in {c_name} (Noun + Answer '{ans}')")
                        
    if classification == "NO_MATCH_FOUND":
        # Check if answer alone appears widely
        for c_name, c_text in training_records[:500]:
            if f" {ans_norm} " in f" {c_text} ":
                classification = "ANSWER_ONLY_OVERLAP"
                evidence.append(f"Answer entity '{ans_norm}' found in {c_name}")
                break
                
    summary[classification] += 1
    results.append({
        "id": p_id,
        "prompt": item["prompt"],
        "expected_answer": item["expected_answer"],
        "classification": classification,
        "evidence": evidence[:3]
    })

output = {
    "total_benchmark_items": len(benchmark),
    "classification_summary": summary,
    "items": results
}

with open("experiments/forensics_nr056/contamination_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("CONTAMINATION AUDIT COMPLETE.")
print(summary)
