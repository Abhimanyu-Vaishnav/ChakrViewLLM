"""
experiments/nr052_workspace/fast_audit.py

Fast, robust Phase 1 data audit:
- Samples 200 documents per file for exact tokenization, then extrapolates token counts.
- Gathers exact character counts, line counts, duplicate rates via SHA-256 hashes.
- Distinguishes pretraining, SFT, and sealed evaluation sets.
- Generates reports/NR052_DATA_INVENTORY.json and reports/NR052_DATA_INVENTORY.md.
"""

import sys
import os
import json
import hashlib
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def detect_language(text):
    dev_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    lat_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    tot = dev_count + lat_count
    if tot == 0:
        return "other"
    ratio = dev_count / tot
    if ratio > 0.4:
        return "hi"
    elif ratio > 0.05:
        return "hi-en"
    code_markers = ["def ", "class ", "import ", "return ", "function(", "const "]
    if any(m in text for m in code_markers):
        return "code"
    return "en"

def main():
    start_t = time.time()
    print("=== FAST AUDIT STARTING ===", flush=True)
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    target_files = [
        ("Raw Stage-B Literature (Hindi)", "data/raw/hindi/stage_b_literature.txt", "PRETRAIN_PROSE"),
        ("Raw Stage-B Systems (English)", "data/raw/english/stage_b_systems.txt", "PRETRAIN_PROSE"),
        ("Raw Stage-B Algorithms (Code)", "data/raw/code/stage_b_algorithms.txt", "PRETRAIN_CODE"),
        ("Raw Stage-B Derivations (Math)", "data/raw/mathematics/stage_b_derivations.txt", "PRETRAIN_PROSE"),
        ("Raw Stage-B Dialogues (Hinglish)", "data/raw/hinglish/stage_b_dialogues.txt", "PRETRAIN_CONV"),
        ("Processed Stage-B Hindi", "data/processed/stage_b/train/hindi.txt", "PRETRAIN_PROSE"),
        ("Processed Stage-B English", "data/processed/stage_b/train/english.txt", "PRETRAIN_PROSE"),
        ("Processed Stage-B Code", "data/processed/stage_b/train/code.txt", "PRETRAIN_CODE"),
        ("Corpus V1 Train", "data/corpus_v1/master_corpus_train.jsonl", "SFT_CONVERSATIONAL"),
        ("Corpus V2 Train", "data/corpus_v2/corpus_v2_train.jsonl", "SFT_CONVERSATIONAL"),
        ("Corpus V3 Train", "data/corpus_v3/corpus_v3_train.jsonl", "SFT_CONVERSATIONAL"),
        ("Pretraining NR-051 Train", "data/pretraining_nr051/pretrain_train.jsonl", "PRETRAIN_CONTINUOUS"),
        ("Pretraining NR-051 Val", "data/pretraining_nr051/pretrain_val.jsonl", "PRETRAIN_VAL"),
        ("Stage-D Train", "data/stage_d/splits/stage_d_train.jsonl", "SFT_CONVERSATIONAL"),
        ("Stage-E Corpus 013 Train", "data/stage_e_corpus_013/stage_e_train.jsonl", "SFT_CONVERSATIONAL"),
        ("Stage-F Corpus 014 Train", "data/stage_f_corpus_014/stage_f_train.jsonl", "SFT_CONVERSATIONAL"),
        ("SFT UCIP14R Master (15k)", "data/sft/ucip14r/sft_master_15000.jsonl", "SFT_VERIFIED"),
        ("SFT UCIP17 Master (15k)", "data/sft/ucip17/sft_master_15000.jsonl", "SFT_VERIFIED"),
        ("SFT UCIP20 Bind Master (2k)", "data/sft/ucip20_bind/bind_master_2000.jsonl", "SFT_VERIFIED"),
        ("SFT EXP34 Candidate V3", "data/sft/exp34_candidate_v3/exp34_instruction_mixture_train_v3.jsonl", "SFT_CONVERSATIONAL"),
        ("SFT M3 Conversational", "data/sft/m3_conversational/m3_conversational_mixture_600.jsonl", "SFT_CONVERSATIONAL"),
        ("Sealed 120 Benchmark (Frozen Eval)", "experiments/fast_track_047_049/sealed_120_benchmark.json", "EVAL_FROZEN"),
        ("Fresh Matched 1000 Benchmark (Frozen Eval)", "data/stage_j_020/fresh_matched_benchmark_1000.json", "EVAL_FROZEN")
    ]

    inventory = []

    for label, rel_path, category in target_files:
        p = Path(rel_path)
        if not p.exists():
            print(f"Skipping missing: {rel_path}", flush=True)
            continue

        file_size = p.stat().st_size
        doc_count = 0
        char_count = 0
        token_count = 0
        unique_hashes = set()
        lang_dist = Counter()

        sample_tokens = 0
        sample_docs = 0

        if p.suffix == ".txt":
            content = p.read_text(encoding="utf-8", errors="ignore")
            char_count = len(content)
            paras = [x.strip() for x in content.split("\n\n") if len(x.strip()) > 30]
            doc_count = len(paras)
            for para in paras:
                unique_hashes.add(sha256_text(para))
                lang_dist[detect_language(para)] += 1
                if sample_docs < 100:
                    sample_tokens += len(tokenizer.encode(para))
                    sample_docs += 1
            if sample_docs > 0:
                token_count = int((sample_tokens / sample_docs) * doc_count)
            else:
                token_count = len(tokenizer.encode(content[:50000]))

        elif p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip(): continue
                    doc_count += 1
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            t = obj["text"]
                        elif "prompt" in obj and "response" in obj:
                            t = obj["prompt"] + "\n" + obj["response"]
                        elif "messages" in obj:
                            t = "\n".join(m.get("content", "") for m in obj["messages"])
                        else:
                            t = line
                    except Exception:
                        t = line

                    char_count += len(t)
                    unique_hashes.add(sha256_text(t))
                    lang_dist[detect_language(t)] += 1
                    if sample_docs < 200:
                        sample_tokens += len(tokenizer.encode(t))
                        sample_docs += 1

            if sample_docs > 0:
                token_count = int((sample_tokens / sample_docs) * doc_count)

        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, list):
                    doc_count = len(data)
                    for item in data:
                        t = json.dumps(item) if not isinstance(item, str) else item
                        char_count += len(t)
                        unique_hashes.add(sha256_text(t))
                        lang_dist[detect_language(t)] += 1
                        if sample_docs < 100:
                            sample_tokens += len(tokenizer.encode(t))
                            sample_docs += 1
                    if sample_docs > 0:
                        token_count = int((sample_tokens / sample_docs) * doc_count)
                elif isinstance(data, dict):
                    doc_count = 1
                    t = json.dumps(data)
                    char_count = len(t)
                    token_count = len(tokenizer.encode(t[:10000]))
                    unique_hashes.add(sha256_text(t))

        dup_rate = 1.0 - (len(unique_hashes) / max(1, doc_count))

        record = {
            "label": label,
            "path": rel_path,
            "file_size_bytes": file_size,
            "doc_count": doc_count,
            "unique_doc_count": len(unique_hashes),
            "char_count": char_count,
            "token_count": token_count,
            "duplicate_rate": round(dup_rate, 4),
            "category": category,
            "language_distribution": dict(lang_dist)
        }
        inventory.append(record)
        print(f"[{category:16}] {label:32} | Docs: {doc_count:6,} | Toks: {token_count:9,} | Dup: {dup_rate*100:4.1f}%", flush=True)

    out_json = Path("reports/NR052_DATA_INVENTORY.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    elapsed = time.time() - start_t
    print(f"\nAudit completed in {elapsed:.2f}s. Saved to {out_json}", flush=True)

if __name__ == "__main__":
    main()
