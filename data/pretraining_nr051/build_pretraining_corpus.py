"""
data/pretraining_nr051/build_pretraining_corpus.py
Constructs an isolated, multi-lingual, multi-domain Pretraining Corpus for NR-051:
- Sources:
  * Raw informational & educational text (data/raw/**/*.txt): English, Hindi, CS, algorithms, math.
  * Curated educational prose extracted from UCIP-14R, UCIP-17, Stage-D/E/F/G/H/I.
  * Python programming code and technical references.
  * Bilingual dialogue and conversational structures.
- Formatting: Unstructured continuous natural language documents separated by <doc> / <eos> boundaries.
  (Crucial: NOT conversational `<user>...<assistant>` prompts, but genuine natural language text for causal LM next-token prediction).
- Deduplication and Benchmark Contamination Gating against sealed benchmarks.
- Produces token manifests and language distributions.
"""

import sys
import os
import json
import glob
import hashlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(text):
    return " ".join(text.strip().split())

def build_pretraining_corpus():
    print("=== BUILDING ISOLATED PRETRAINING CORPUS (NR-051) ===")
    
    out_dir = Path("data/pretraining_nr051")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    # 1. Contamination check against sealed benchmarks
    sealed_files = [
        "experiments/fast_track_047_049/sealed_120_benchmark.json",
        "data/stage_j_020/fresh_matched_benchmark_1000.json"
    ]
    benchmark_prompts = set()
    for sf in sealed_files:
        if os.path.exists(sf):
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    benchmark_prompts.add(normalize_text(item.get("prompt", "")))
    print(f"Loaded {len(benchmark_prompts)} sealed benchmark prompts for contamination gating.")

    documents = []
    seen_hashes = set()
    lang_counter = Counter()
    domain_counter = Counter()

    def add_doc(text, lang, domain, source_id):
        text = text.strip()
        if len(text) < 20:
            return
        # Deduplication via text hash
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in seen_hashes:
            return
        seen_hashes.add(doc_hash)

        # Contamination check
        norm = normalize_text(text)
        for bp in benchmark_prompts:
            if len(bp) > 20 and bp in norm:
                return

        documents.append({
            "text": text,
            "language": lang,
            "domain": domain,
            "source": source_id
        })
        lang_counter[lang] += 1
        domain_counter[domain] += 1

    # Source 1: Raw Educational & Technical Text Files
    print("Processing data/raw/ text files...")
    for txt_path in glob.glob("data/raw/**/*.txt", recursive=True):
        lang = "en"
        if "hindi" in txt_path.lower(): lang = "hi"
        elif "sanskrit" in txt_path.lower(): lang = "sa"
        elif "mixed" in txt_path.lower() or "bilingual" in txt_path.lower(): lang = "hinglish"

        domain = "technical"
        if "code" in txt_path.lower(): domain = "code"
        elif "math" in txt_path.lower() or "numbers" in txt_path.lower(): domain = "math"
        elif "systems" in txt_path.lower() or "algorithms" in txt_path.lower(): domain = "algorithms"
        elif "dialogue" in txt_path.lower(): domain = "conversational"

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Split into paragraphs/sections
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
            for p in paragraphs:
                add_doc(p, lang, domain, txt_path)

    # Source 2: Educational Explanations from UCIP-14R & UCIP-17
    print("Processing UCIP curated informational datasets...")
    for ucip_path in ["data/sft/ucip14r/sft_train_12000.jsonl", "data/sft/ucip17/sft_train_12000.jsonl"]:
        if not os.path.exists(ucip_path):
            continue
        with open(ucip_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cat = d.get("category", "")
                # Skip copy tasks and raw format echoing
                if "copy" in cat.lower() or cat in ["COPY", "FORMAT"]:
                    continue
                
                lang = d.get("language", "en")
                if lang in ["en", "ENG"]: lang = "en"
                elif lang in ["hi", "HIN", "hindi"]: lang = "hi"
                elif lang in ["hi-en", "hinglish", "HING"]: lang = "hinglish"
                elif lang in ["sa", "SAN"]: lang = "sa"

                prompt = d.get("prompt", "").replace("User:", "").replace("Assistant:", "").strip()
                response = d.get("response", "").strip()

                # Convert Q&A pair into a continuous expository document
                doc_text = f"{prompt} {response}"
                add_doc(doc_text, lang, "informational_qa", ucip_path)

    # Source 3: Code and Algorithm Datasets (Stage E, F, Stage D)
    print("Processing code and algorithm corpora...")
    for stage_path in ["data/stage_e_corpus_013/stage_e_train.jsonl", "data/stage_d/splits/stage_d_train.jsonl"]:
        if not os.path.exists(stage_path):
            continue
        with open(stage_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                p = d.get("prompt", "").replace("User:", "").replace("Assistant:", "").strip()
                r = d.get("response", "").strip()
                add_doc(f"{p}\n{r}", "en", "code_and_algorithms", stage_path)

    print(f"Total Unique Extracted Documents: {len(documents):,}")
    print(f"Languages: {dict(lang_counter)}")
    print(f"Domains: {dict(domain_counter)}")

    # 4. Split into Train (95%) and Validation (5%)
    val_count = int(len(documents) * 0.05)
    train_docs = documents[:-val_count]
    val_docs = documents[-val_count:]

    train_path = out_dir / "pretrain_train.jsonl"
    val_path = out_dir / "pretrain_val.jsonl"

    train_tokens = 0
    val_tokens = 0

    print("Writing pretraining splits and computing token counts...")
    with open(train_path, "w", encoding="utf-8") as f:
        for d in train_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            train_tokens += len(tokenizer.encode(d["text"]))

    with open(val_path, "w", encoding="utf-8") as f:
        for d in val_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            val_tokens += len(tokenizer.encode(d["text"]))

    manifest = {
        "corpus_name": "ChakrView Pretraining Corpus (NR-051)",
        "total_documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens_unique": train_tokens,
        "val_tokens_unique": val_tokens,
        "total_tokens_unique": train_tokens + val_tokens,
        "languages": dict(lang_counter),
        "domains": dict(domain_counter),
        "files": {
            "train": {
                "path": str(train_path),
                "sha256": file_sha256(train_path),
                "size_bytes": os.path.getsize(train_path),
                "tokens": train_tokens
            },
            "val": {
                "path": str(val_path),
                "sha256": file_sha256(val_path),
                "size_bytes": os.path.getsize(val_path),
                "tokens": val_tokens
            }
        }
    }

    manifest_path = out_dir / "pretrain_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== PRETRAINING CORPUS GENERATION COMPLETE ===")
    print(f"Unique Documents: {len(documents):,}")
    print(f"Train Tokens:     {train_tokens:,}")
    print(f"Val Tokens:       {val_tokens:,}")
    print(f"Total Tokens:     {train_tokens + val_tokens:,}")
    print(f"Train SHA-256:    {manifest['files']['train']['sha256']}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    build_pretraining_corpus()
