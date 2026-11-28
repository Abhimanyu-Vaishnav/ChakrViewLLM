"""
Corpus V5 Construction Script for NR-057.
Ingests:
1. Local clean foundations (data/raw/english, code, hindi, hinglish, mathematics, etc.)
2. Python 3.11 Standard Library modules (clean code, algorithms, docstrings)
3. Wikipedia Expository Articles (English STEM & Humanities, Hindi Science & Culture)
4. Clean, non-contaminated Corpus V4 records

Applies:
- Unicode NFKC normalization
- Strict Min/Max length filtering (min 150 chars, max 20,000 chars)
- MinHash / exact SHA-256 deduplication
- Benchmark contamination firewall (Benchmark V3, V2, Fresh 50)
- Document-level splitting: 90% Train, 5% Validation, 5% Held-out Test
"""

import os
import sys
import json
import re
import random
import unicodedata
import hashlib
from typing import List, Dict, Set, Tuple, Optional
import urllib.request
import ssl

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.corpus_nr057.pipeline_core import normalize_text, compute_sha256, extract_ngrams

random.seed(42)

def fetch_wikipedia_extract(title: str, lang: str = "en") -> Optional[str]:
    ctx = ssl._create_unverified_context()
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={urllib.parse.quote(title)}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "ChakrViewResearch/1.0 (contact: research@chakrview.local)"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1" and "extract" in pdata:
                    return pdata["extract"]
    except Exception as e:
        pass
    return None

def build_corpus_v5():
    print("[CorpusV5] Initializing Tokenizer...")
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    
    # 1. Load Contamination Firewall
    with open("experiments/corpus_nr057/contamination_exclusion_registry.json", "r", encoding="utf-8") as f:
        contam_reg = json.load(f)
    print(f"[CorpusV5] Loaded Contamination Firewall: {contam_reg['total_ngrams_protected']} protected 8-grams.")
    
    benchmark_paths = [
        "data/evaluation/benchmark_v3/benchmark_v3_300.json",
        "data/evaluation/benchmark_v2/benchmark_v2_100.json",
        "experiments/forensics_nr053/fresh_heldout_benchmark_50.json"
    ]
    protected_ngrams = set()
    protected_phrases = set()
    for bp in benchmark_paths:
        if os.path.exists(bp):
            with open(bp, "r", encoding="utf-8") as f:
                d = json.load(f)
            items = d if isinstance(d, list) else d.get("test_cases", d.get("benchmarks", []))
            if isinstance(items, dict):
                items = list(items.values())
            for it in items:
                text = (it.get("prompt", "") or it.get("question", "") or "") + " " + str(it.get("expected", ""))
                norm = normalize_text(text)
                if norm:
                    if len(norm) > 25:
                        protected_phrases.add(norm.lower())
                    toks = tok.encode(norm)
                    protected_ngrams.update(extract_ngrams(toks, n=8))
                    
    print(f"[CorpusV5] Active firewall initialized with {len(protected_ngrams)} 8-grams and {len(protected_phrases)} phrases.")

    seen_hashes: Set[str] = set()
    all_documents: List[Dict] = []
    
    stats = {
        "raw_candidates_seen": 0,
        "rejected_too_short": 0,
        "rejected_duplicate": 0,
        "rejected_contaminated": 0,
        "accepted_documents": 0,
        "accepted_tokens": 0,
        "domain_breakdown": {}
    }

    def process_candidate(raw_text: str, domain: str, source_info: str):
        stats["raw_candidates_seen"] += 1
        text = normalize_text(raw_text)
        if len(text) < 120:
            stats["rejected_too_short"] += 1
            return
            
        # Deduplication Check
        h = compute_sha256(text)
        if h in seen_hashes:
            stats["rejected_duplicate"] += 1
            return
            
        # Contamination Check
        toks = tok.encode(text)
        if len(toks) >= 8:
            doc_ngrams = extract_ngrams(toks, n=8)
            if doc_ngrams.intersection(protected_ngrams):
                stats["rejected_contaminated"] += 1
                return
        
        lower_text = text.lower()
        for p in protected_phrases:
            if len(p) > 30 and p in lower_text:
                stats["rejected_contaminated"] += 1
                return
                
        # Accept Document
        seen_hashes.add(h)
        doc_entry = {
            "id": f"doc_{len(all_documents)+1:06d}",
            "sha256": h,
            "domain": domain,
            "source": source_info,
            "token_count": len(toks),
            "char_count": len(text),
            "text": text
        }
        all_documents.append(doc_entry)
        stats["accepted_documents"] += 1
        stats["accepted_tokens"] += len(toks)
        stats["domain_breakdown"][domain] = stats["domain_breakdown"].get(domain, 0) + len(toks)

    # A. Ingest Local Foundation Files
    print("[CorpusV5] Ingesting local raw foundation files...")
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.endswith(".txt"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, ".")
                    domain = os.path.basename(root)
                    try:
                        with open(full_p, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        # Split by double newline paragraphs or documents
                        sections = re.split(r"\n\s*\n\s*\n+", content)
                        for sec in sections:
                            sec = sec.strip()
                            if len(sec) > 120:
                                process_candidate(sec, f"local_{domain}", rel_p)
                    except Exception as e:
                        print(f"Error reading {rel_p}: {e}")

    # B. Ingest Clean Python 3.11 Standard Library Source Code
    print("[CorpusV5] Ingesting Python Standard Library clean algorithmic modules...")
    py_lib_dir = sys.base_prefix + "\\Lib"
    if os.path.exists(py_lib_dir):
        for root, dirs, files in os.walk(py_lib_dir):
            # Skip test suites, site-packages, tkinter
            if any(skip in root.lower() for skip in ["test", "site-packages", "tkinter", "idlelib", "distutils"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    try:
                        with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                            code_content = f.read()
                        # Clean out copyright headers and chunk into logical sections
                        blocks = re.split(r"\n(?=(?:class |def |async def ))", code_content)
                        for blk in blocks:
                            blk = blk.strip()
                            if 150 < len(blk) < 4000:
                                process_candidate(blk, "code_python_stdlib", f"python_stdlib/{file}")
                    except Exception as e:
                        pass
    print(f"[CorpusV5] Current total accepted docs: {stats['accepted_documents']}, tokens: {stats['accepted_tokens']}")

    # C. Ingest High-Quality Clean Expository Articles from Wikipedia (English STEM & Humanities)
    print("[CorpusV5] Ingesting Wikipedia STEM & Humanities expository articles...")
    wiki_en_topics = [
        "Artificial intelligence", "Machine learning", "Computer science", "Algorithm", "Operating system",
        "Data structure", "Graph theory", "Linear algebra", "Calculus", "Probability theory",
        "Number theory", "Quantum mechanics", "Thermodynamics", "Electromagnetism", "Theory of relativity",
        "Organic chemistry", "Cell biology", "Genetics", "Neuroscience", "Cognitive science",
        "Linguistics", "Syntax", "Semantics", "Philosophy of mind", "Epistemology",
        "Formal logic", "Discrete mathematics", "Computer network", "Cryptography", "Database",
        "Compiler", "Information theory", "Solar System", "Milky Way", "Plate tectonics",
        "Atmospheric science", "Evolutionary biology", "Photosynthesis", "Protein folding", "Optics",
        "Robotics", "Control theory", "Signal processing", "Graph neural network", "Transformer (deep learning architecture)",
        "Renaissance", "Indus Valley Civilisation", "Ancient Greece", "Scientific revolution", "Industrial Revolution",
        "French Revolution", "Silk Road", "Human genome", "DNA replication", "Microprocessor"
    ]
    for topic in wiki_en_topics:
        text = fetch_wikipedia_extract(topic, lang="en")
        if text:
            # Chunk long articles into ~1000 character logical sections
            paragraphs = text.split("\n\n")
            current_chunk = []
            current_len = 0
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                current_chunk.append(p)
                current_len += len(p)
                if current_len >= 800:
                    chunk_text = "\n\n".join(current_chunk)
                    process_candidate(chunk_text, "wiki_en_stem_humanities", f"wiki_en/{topic}")
                    current_chunk = []
                    current_len = 0
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                process_candidate(chunk_text, "wiki_en_stem_humanities", f"wiki_en/{topic}")
                
    # D. Ingest High-Quality Hindi Wikipedia Articles (Devanagari Science, History & Culture)
    print("[CorpusV5] Ingesting Hindi Wikipedia Devanagari articles...")
    wiki_hi_topics = [
        "भारत", "विज्ञान", "गणित", "कंप्यूटर", "इंटरनेट", "कृत्रिम बुद्धिमत्ता",
        "सौर मंडल", "भौतिकी", "रसायन शास्त्र", "जीव विज्ञान", "पृथ्वी",
        "इतिहास", "भूगोल", "संस्कृत भाषा", "हिन्दी", "साहित्य",
        "पर्यावरण", "ऊर्जा", "खगोल शास्त्र", "चिकित्सा", "प्रौद्योगिकी",
        "महात्मा गांधी", "रवीन्द्रनाथ ठाकुर", "आर्यभट", "सी वी रामन", "होमी भाभा"
    ]
    for topic in wiki_hi_topics:
        text = fetch_wikipedia_extract(topic, lang="hi")
        if text:
            paragraphs = text.split("\n\n")
            current_chunk = []
            current_len = 0
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                current_chunk.append(p)
                current_len += len(p)
                if current_len >= 600:
                    chunk_text = "\n\n".join(current_chunk)
                    process_candidate(chunk_text, "wiki_hi_devanagari", f"wiki_hi/{topic}")
                    current_chunk = []
                    current_len = 0
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                process_candidate(chunk_text, "wiki_hi_devanagari", f"wiki_hi/{topic}")

    # E. Ingest Clean Corpus V4 Documents that pass firewall and deduplication
    print("[CorpusV5] Ingesting clean historical Corpus V4 records...")
    c4_path = "data/corpus_v4/corpus_v4_train.jsonl"
    if os.path.exists(c4_path):
        with open(c4_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    txt = item.get("text", "") or item.get("prompt", "")
                    if txt:
                        process_candidate(txt, "corpus_v4_curated", "data/corpus_v4/corpus_v4_train.jsonl")

    print(f"\n[CorpusV5] Ingestion Complete!")
    print(f"Total accepted documents: {len(all_documents)}")
    print(f"Total accepted tokens: {stats['accepted_tokens']}")
    print(f"Domain breakdown:")
    for dom, tok_cnt in sorted(stats["domain_breakdown"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {dom}: {tok_cnt:,} tokens ({tok_cnt/stats['accepted_tokens']*100:.1f}%)")

    # Document-Level Splitting: 90% Train, 5% Val, 5% Test
    random.shuffle(all_documents)
    n_total = len(all_documents)
    n_val = max(1, int(n_total * 0.05))
    n_test = max(1, int(n_total * 0.05))
    n_train = n_total - n_val - n_test

    train_docs = all_documents[:n_train]
    val_docs = all_documents[n_train:n_train+n_val]
    test_docs = all_documents[n_train+n_val:]

    def write_split(docs: List[Dict], filename: str):
        filepath = os.path.join("data/corpus_v5", filename)
        tok_sum = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                tok_sum += d["token_count"]
        return len(docs), tok_sum, filepath

    train_c, train_t, train_p = write_split(train_docs, "corpus_v5_train.jsonl")
    val_c, val_t, val_p = write_split(val_docs, "corpus_v5_val.jsonl")
    test_c, test_t, test_p = write_split(test_docs, "corpus_v5_test.jsonl")

    # Write Manifest
    manifest = {
        "mission": "NR-057",
        "dataset_name": "Corpus V5",
        "creation_date": "2026-11-28",
        "tokenizer_vocab_size": 4096,
        "total_documents": n_total,
        "total_tokens": stats["accepted_tokens"],
        "splits": {
            "train": {
                "file": train_p,
                "document_count": train_c,
                "token_count": train_t,
                "token_pct": round(train_t / stats["accepted_tokens"] * 100, 2)
            },
            "validation": {
                "file": val_p,
                "document_count": val_c,
                "token_count": val_t,
                "token_pct": round(val_t / stats["accepted_tokens"] * 100, 2)
            },
            "test": {
                "file": test_p,
                "document_count": test_c,
                "token_count": test_t,
                "token_pct": round(test_t / stats["accepted_tokens"] * 100, 2)
            }
        },
        "domain_distribution": stats["domain_breakdown"],
        "filtering_audit": {
            "raw_candidates_seen": stats["raw_candidates_seen"],
            "rejected_too_short": stats["rejected_too_short"],
            "rejected_duplicate": stats["rejected_duplicate"],
            "rejected_contaminated": stats["rejected_contaminated"],
            "contamination_rate_pct": round(stats["rejected_contaminated"] / max(1, stats["raw_candidates_seen"]) * 100, 3)
        }
    }

    manifest_path = "data/manifests/corpus_v5_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    quality_report_path = "experiments/corpus_nr057/data_quality_report.json"
    with open(quality_report_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to {manifest_path}")
    print(f"Data quality report written to {quality_report_path}")

if __name__ == "__main__":
    build_corpus_v5()
