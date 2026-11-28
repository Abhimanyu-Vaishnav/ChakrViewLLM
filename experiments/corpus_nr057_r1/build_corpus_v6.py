"""
Corpus V6 Construction Script for NR-057-R1.
Target: Rebalance Corpus V5 and expand towards clean usable token budget.
Prioritizes:
1. Native Hindi educational prose (expanded Wikipedia Hindi articles & Devanagari literature)
2. Natural Hinglish dialogues and conversational archives
3. English educational prose (expanded STEM, history, philosophy, geography)
4. Formal mathematics, arithmetic proofs, and scientific derivations
5. Structured logic and reasoning procedures

Strict rules:
- Zero external LLM APIs or synthetic outputs
- Zero pretrained weights
- Canonical Byte-Level BPE 4096 tokenizer
- Dual-layer Benchmark Contamination Firewall V2
- Document-level split (90% Train, 5% Val, 5% Test)
- Generates corpus_v6_train.jsonl, corpus_v6_val.jsonl, corpus_v6_test.jsonl
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

def build_corpus_v6():
    print("[CorpusV6] Initializing Tokenizer...")
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    
    # 1. Load Contamination Firewall V2
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
                    
    print(f"[CorpusV6] Firewall V2 loaded with {len(protected_ngrams)} 8-grams and {len(protected_phrases)} phrases.")

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
            
        # Contamination Firewall V2
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
            "id": f"v6_doc_{len(all_documents)+1:06d}",
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

    # A. Ingest All Clean Documents from Corpus V5 (Preserve clean foundation)
    print("[CorpusV6] Re-ingesting clean Corpus V5 foundation records...")
    for s_name in ["train", "val", "test"]:
        v5_p = f"data/corpus_v5/corpus_v5_{s_name}.jsonl"
        if os.path.exists(v5_p):
            with open(v5_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        process_candidate(item["text"], item["domain"], item["source"])
    print(f"[CorpusV6] After Corpus V5 base: {stats['accepted_documents']} docs, {stats['accepted_tokens']:,} tokens.")

    # B. Expanded Hindi Devanagari Expository Articles
    print("[CorpusV6] Ingesting expanded Hindi Devanagari encyclopedic articles...")
    wiki_hi_expanded = [
        "संविधान", "लोकतंत्र", "भारतीय राष्ट्रीय आन्दोलन", "मौलिक अधिकार", "अर्थशास्त्र",
        "भौतिक विज्ञान", "रसायन विज्ञान", "खगोलिकी", "जीवविज्ञान", "पारिस्थितिकी",
        "गंगा", "हिमालय", "सिंधु घाटी सभ्यता", "मौर्य साम्राज्य", "गुप्त साम्राज्य",
        "मुगल साम्राज्य", "दिल्ली", "मुम्बई", "कोलकाता", "चेन्नई", "बेंगलुरु",
        "सूर्य", "चन्द्रमा", "मंगल ग्रह", "बृहस्पति ग्रह", "गुरुत्वाकर्षण",
        "प्रकाशिकी", "विद्युत चुम्बकत्व", "ऊष्मागतिकी", "नाभिकीय भौतिकी",
        "कम्प्यूटर विज्ञान", "एल्गोरिद्म", "प्रोग्रामन भाषा", "सॉफ्टवेयर", "डेटाबेस",
        "प्रेमचंद", "जयशंकर प्रसाद", "तुलसीदास", "कबीर", "कालिदास",
        "महाभारत", "रामायण", "उपनिषद्", "वेद", "आयुर्वेद",
        "कृषि", "जलवायु परिवर्तन", "जल संरक्षण", "नवीकरणीय ऊर्जा", "जैव विविधता"
    ]
    for topic in wiki_hi_expanded:
        text = fetch_wikipedia_extract(topic, lang="hi")
        if text:
            paras = text.split("\n\n")
            cur = []
            cur_len = 0
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                cur.append(p)
                cur_len += len(p)
                if cur_len >= 600:
                    process_candidate("\n\n".join(cur), "wiki_hi_devanagari", f"wiki_hi/{topic}")
                    cur = []
                    cur_len = 0
            if cur:
                process_candidate("\n\n".join(cur), "wiki_hi_devanagari", f"wiki_hi/{topic}")

    # C. Expanded Hinglish Natural Conversational Archives
    print("[CorpusV6] Ingesting natural Hinglish conversational and technical dialogues...")
    # Check data/raw/hinglish and other curated conversational archives
    for root, _, files in os.walk("data/raw/hinglish"):
        for file in files:
            if file.endswith(".txt"):
                p = os.path.join(root, file)
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        for chunk in re.split(r"\n\s*\n+", f.read()):
                            if len(chunk.strip()) > 100:
                                process_candidate(chunk.strip(), "hinglish_natural_dialogue", f"raw_hinglish/{file}")
                except Exception as e:
                    pass

    # D. Expanded English STEM & Humanities Expository Articles
    print("[CorpusV6] Ingesting expanded English STEM & Humanities articles...")
    wiki_en_expanded = [
        "Thermodynamics", "Electromagnetism", "Special relativity", "General relativity", "Quantum field theory",
        "Atomic physics", "Statistical mechanics", "Fluid dynamics", "Planetary science", "Astrophysics",
        "Cell (biology)", "Molecular biology", "Biochemistry", "Genetics", "Evolutionary developmental biology",
        "Immune system", "Central nervous system", "Cognitive psychology", "Philosophy of science", "Logic",
        "Set theory", "Combinatorics", "Topology", "Complex analysis", "Numerical analysis",
        "Distributed computing", "Concurrency (computer science)", "Computer architecture", "Network protocol",
        "Operating system kernel", "Memory management", "Relational database", "Object-oriented programming",
        "Functional programming", "Abstract data type", "Computational complexity theory", "Automata theory",
        "History of mathematics", "History of science", "Age of Enlightenment", "Printing press",
        "Microbiology", "Ecology", "Climate change", "Renewable energy", "Semiconductor device fabrication"
    ]
    for topic in wiki_en_expanded:
        text = fetch_wikipedia_extract(topic, lang="en")
        if text:
            paras = text.split("\n\n")
            cur = []
            cur_len = 0
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                cur.append(p)
                cur_len += len(p)
                if cur_len >= 800:
                    process_candidate("\n\n".join(cur), "wiki_en_stem_humanities", f"wiki_en/{topic}")
                    cur = []
                    cur_len = 0
            if cur:
                process_candidate("\n\n".join(cur), "wiki_en_stem_humanities", f"wiki_en/{topic}")

    # E. Expanded Formal Mathematics & Logic
    print("[CorpusV6] Ingesting formal mathematics and reasoning procedures...")
    math_dir = "data/raw/mathematics"
    if os.path.exists(math_dir):
        for root, _, files in os.walk(math_dir):
            for file in files:
                if file.endswith(".txt"):
                    p = os.path.join(root, file)
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            for chunk in re.split(r"\n\s*\n+", f.read()):
                                if len(chunk.strip()) > 80:
                                    process_candidate(chunk.strip(), "formal_mathematics_logic", f"raw_math/{file}")
                    except Exception as e:
                        pass

    print(f"\n[CorpusV6] Construction complete!")
    print(f"Total accepted documents: {len(all_documents)}")
    print(f"Total accepted tokens: {stats['accepted_tokens']:,}")
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
        filepath = os.path.join("data/corpus_v6/splits", filename)
        tok_sum = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                tok_sum += d["token_count"]
        return len(docs), tok_sum, filepath

    train_c, train_t, train_p = write_split(train_docs, "corpus_v6_train.jsonl")
    val_c, val_t, val_p = write_split(val_docs, "corpus_v6_val.jsonl")
    test_c, test_t, test_p = write_split(test_docs, "corpus_v6_test.jsonl")

    # Write Manifest
    manifest = {
        "mission": "NR-057-R1",
        "dataset_name": "Corpus V6",
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

    manifest_path = "data/manifests/corpus_v6_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to {manifest_path}")

if __name__ == "__main__":
    build_corpus_v6()
