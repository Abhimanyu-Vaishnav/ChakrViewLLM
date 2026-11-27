import os
import sys
import json
import re
import hashlib
from collections import Counter
from pathlib import Path

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(text):
    return " ".join(text.strip().split())

def parse_user_prompt(prompt_str, messages=None):
    """
    Explicit parser to extract the genuine user instruction:
    1. If structured 'messages' list is available, extract content from role == 'user'.
    2. If string starts with 'User:', extract the substring up to '\nAssistant:' or 'Assistant:'.
    3. Handles multi-turn or trailing whitespace cleanly.
    """
    if messages and isinstance(messages, list):
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if user_msgs:
            return user_msgs[0].strip()

    p = prompt_str.strip()
    if p.startswith("User:"):
        p = p[5:].strip()
    if p.endswith("Assistant:"):
        p = p[:-10].strip()
    
    # Check if there is an embedded Assistant: tag
    if "\nAssistant:" in p:
        p = p.split("\nAssistant:")[0].strip()
    elif "Assistant:" in p:
        p = p.split("Assistant:")[0].strip()
        
    return p

def classify_copy_task(prompt, response, category):
    """
    Classify copy tasks rather than blindly dropping them:
    - If task explicitly asks to copy a synthetic hash/token and response is 1-2 words -> SYNTHETIC_TOKEN_COPY
    - If task is short but conversational (e.g. greeting, yes/no, clarification) -> CONVERSATIONAL_SHORT
    - If task is instruction/formatting -> FORMAT_COMPLIANCE
    """
    p_lower = prompt.lower()
    cat_lower = category.lower()
    resp_words = response.strip().split()
    
    if "copy the following token" in p_lower or "exact copy" in p_lower or "ठीक वैसे ही लिखें" in prompt or "यथातथं प्रतिलिखतु" in prompt:
        return "SYNTHETIC_TOKEN_COPY"
    
    if "copy" in cat_lower:
        if len(resp_words) <= 2:
            return "SYNTHETIC_TOKEN_COPY"
        else:
            return "STRUCTURED_COPY"
            
    return category

def build_corpus_v2():
    print("=== BUILDING CHAKRVIEW 1.0 CORPUS V2 (CONTROLLED REMEDIATION) ===")
    
    out_dir = Path("data/corpus_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Contamination Prevention against sealed benchmarks
    sealed_bmk_path = Path("data/stage_j_020/fresh_matched_benchmark_1000.json")
    benchmark_prompts = set()
    if sealed_bmk_path.exists():
        with open(sealed_bmk_path, "r", encoding="utf-8") as f:
            bmk_data = json.load(f)
            for item in bmk_data:
                benchmark_prompts.add(normalize_text(item["prompt"]))
    print(f"Loaded {len(benchmark_prompts)} sealed benchmark prompts for contamination gating.")

    # 2. Raw Sources (preserve Corpus V1 and read from verified sources)
    sources = [
        {"path": "data/sft/ucip14r/sft_train_12000.jsonl", "split": "train"},
        {"path": "data/sft/ucip14r/sft_validation_1500.jsonl", "split": "val"},
        {"path": "data/stage_d/splits/stage_d_train.jsonl", "split": "train"},
        {"path": "data/stage_d/splits/stage_d_val.jsonl", "split": "val"},
        {"path": "data/sft/exp34_instruction_mixture_train.jsonl", "split": "train"}
    ]

    seen_prompts = set()
    train_records = []
    val_records = []
    
    stats = {
        "total_read": 0,
        "synthetic_copy_filtered": 0,
        "synthetic_copy_retained": 0,
        "short_conversational_preserved": 0,
        "nested_prompts_repaired": 0,
        "duplicates_skipped": 0,
        "contamination_skipped": 0,
        "categories": Counter(),
        "languages": Counter()
    }

    # We retain a balanced, small subset of synthetic copy (e.g. up to 100) to preserve token-copying capability without dominating the loss
    MAX_SYNTHETIC_COPY_RETAINED = 120
    synthetic_copy_retained_count = 0

    for src in sources:
        p = Path(src["path"])
        if not p.exists():
            print(f"Warning: source {p} not found, skipping.")
            continue

        print(f"Processing source: {p} ({src['split']})...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["total_read"] += 1
                d = json.loads(line)
                
                raw_prompt = d.get("prompt", "")
                raw_response = d.get("response", "")
                messages = d.get("messages", None)
                
                if not raw_prompt or not raw_response:
                    continue

                cleaned_prompt = parse_user_prompt(raw_prompt, messages=messages)
                if cleaned_prompt != raw_prompt:
                    stats["nested_prompts_repaired"] += 1

                norm_p = normalize_text(cleaned_prompt)

                # Contamination check
                if norm_p in benchmark_prompts:
                    stats["contamination_skipped"] += 1
                    continue

                # Deduplication check
                if norm_p in seen_prompts:
                    stats["duplicates_skipped"] += 1
                    continue
                seen_prompts.add(norm_p)

                category = d.get("category", "GENERAL")
                sub_category = classify_copy_task(cleaned_prompt, raw_response, category)

                # Handle COPY classification
                if sub_category == "SYNTHETIC_TOKEN_COPY":
                    if synthetic_copy_retained_count < MAX_SYNTHETIC_COPY_RETAINED:
                        synthetic_copy_retained_count += 1
                        stats["synthetic_copy_retained"] += 1
                    else:
                        stats["synthetic_copy_filtered"] += 1
                        continue
                elif len(raw_response.strip().split()) <= 4:
                    # Meaningful short responses (e.g. greetings, yes/no, simple facts) are preserved
                    stats["short_conversational_preserved"] += 1

                lang = d.get("language", "en")
                if lang in ["en", "ENG"]: lang = "en"
                elif lang in ["hi", "HIN", "hindi"]: lang = "hi"
                elif lang in ["hi-en", "hinglish", "HING"]: lang = "hinglish"
                elif lang in ["sa", "SAN", "sanskrit"]: lang = "sa"

                record = {
                    "id": d.get("id", f"CBR20-{len(seen_prompts):06d}"),
                    "prompt": cleaned_prompt,
                    "response": raw_response.strip(),
                    "language": lang,
                    "category": category,
                    "sub_category": sub_category,
                    "source": str(p),
                    "raw_prompt": raw_prompt
                }

                stats["categories"][category] += 1
                stats["languages"][lang] += 1

                if src["split"] == "val":
                    val_records.append(record)
                else:
                    train_records.append(record)

    train_path = out_dir / "corpus_v2_train.jsonl"
    val_path = out_dir / "corpus_v2_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "corpus_name": "ChakrView 1.0 Controlled Remediation Corpus V2",
        "total_unique_records": len(train_records) + len(val_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "stats": {
            "total_read": stats["total_read"],
            "duplicates_skipped": stats["duplicates_skipped"],
            "contamination_skipped": stats["contamination_skipped"],
            "nested_prompts_repaired": stats["nested_prompts_repaired"],
            "synthetic_copy_filtered": stats["synthetic_copy_filtered"],
            "synthetic_copy_retained": stats["synthetic_copy_retained"],
            "short_conversational_preserved": stats["short_conversational_preserved"]
        },
        "languages": dict(stats["languages"]),
        "categories": dict(stats["categories"].most_common(30)),
        "files": {
            "train": {
                "path": str(train_path),
                "sha256": file_sha256(train_path),
                "size_bytes": os.path.getsize(train_path)
            },
            "val": {
                "path": str(val_path),
                "sha256": file_sha256(val_path),
                "size_bytes": os.path.getsize(val_path)
            }
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== CORPUS V2 BUILD COMPLETE ===")
    print(f"Train records: {len(train_records)}")
    print(f"Val records:   {len(val_records)}")
    print(f"Repaired nested prompts: {stats['nested_prompts_repaired']}")
    print(f"Synthetic copy tasks filtered: {stats['synthetic_copy_filtered']} (retained balanced: {stats['synthetic_copy_retained']})")
    print(f"Short conversational responses preserved: {stats['short_conversational_preserved']}")
    print(f"Train SHA-256: {manifest['files']['train']['sha256']}")
    print(f"Val SHA-256:   {manifest['files']['val']['sha256']}")

if __name__ == "__main__":
    build_corpus_v2()
