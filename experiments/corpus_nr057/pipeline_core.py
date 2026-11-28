"""
Ingestion, Normalization, Deduplication, and Benchmark Contamination Exclusion Pipeline for NR-057 (Corpus V5).
"""

import os
import sys
import json
import re
import unicodedata
import hashlib
from typing import List, Dict, Set, Tuple, Optional
import urllib.request
import ssl

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def get_bpe_tokenizer():
    tok_dir = "data/experiments/vocab_4096"
    if not os.path.exists(tok_dir):
        raise FileNotFoundError(f"Tokenizer directory not found: {tok_dir}")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    return tokenizer

def normalize_text(text: str) -> str:
    """Normalize text using Unicode NFKC and strip excessive whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except standard whitespace \n, \t, \r
    text = "".join(ch for ch in text if ch in "\n\t\r" or not unicodedata.category(ch).startswith("C"))
    # Replace multiple spaces with single space, preserve single newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Remove excessive blank lines (more than 2 consecutive newlines)
    cleaned_lines = []
    blank_count = 0
    for l in lines:
        if not l:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(l)
    return "\n".join(cleaned_lines).strip()

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_ngrams(tokens: List[int], n: int = 10) -> Set[Tuple[int, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

class BenchmarkContaminationAuditor:
    """Audits documents against benchmark test sets using exact n-gram matching."""
    def __init__(self, tokenizer: BpeTokenizer, benchmark_files: List[str], n: int = 8):
        self.tokenizer = tokenizer
        self.n = n
        self.benchmark_ngrams: Set[Tuple[int, ...]] = set()
        self.benchmark_phrases: Set[str] = set()
        self._load_benchmarks(benchmark_files)

    def _load_benchmarks(self, benchmark_files: List[str]):
        print(f"[BenchmarkContaminationAuditor] Loading benchmark test sets for contamination filtering...")
        count = 0
        for path in benchmark_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if path.endswith(".json"):
                    data = json.loads(content)
                    items = data if isinstance(data, list) else data.get("test_cases", data.get("benchmarks", data.get("samples", [])))
                    if isinstance(items, dict):
                        items = list(items.values())
                    for item in items:
                        text_to_check = ""
                        if isinstance(item, dict):
                            text_to_check = item.get("prompt", "") + " " + item.get("question", "") + " " + str(item.get("expected", ""))
                        elif isinstance(item, str):
                            text_to_check = item
                        if text_to_check.strip():
                            norm = normalize_text(text_to_check)
                            self.benchmark_phrases.add(norm.lower())
                            toks = self.tokenizer.encode(norm)
                            self.benchmark_ngrams.update(extract_ngrams(toks, self.n))
                            count += 1
                elif path.endswith(".jsonl"):
                    for line in content.splitlines():
                        if line.strip():
                            item = json.loads(line)
                            t = item.get("prompt", "") or item.get("text", "") or item.get("instruction", "")
                            if t.strip():
                                norm = normalize_text(t)
                                self.benchmark_phrases.add(norm.lower())
                                toks = self.tokenizer.encode(norm)
                                self.benchmark_ngrams.update(extract_ngrams(toks, self.n))
                                count += 1
            except Exception as e:
                print(f"Warning: could not process benchmark file {path}: {e}")
        print(f"[BenchmarkContaminationAuditor] Loaded {count} benchmark items. Extracted {len(self.benchmark_ngrams)} distinct {self.n}-grams.")

    def is_contaminated(self, doc_tokens: List[int], doc_text: str) -> Tuple[bool, str]:
        if len(doc_tokens) >= self.n:
            doc_ngrams = extract_ngrams(doc_tokens, self.n)
            overlap = doc_ngrams.intersection(self.benchmark_ngrams)
            if overlap:
                return True, f"Overlap on {len(overlap)} {self.n}-grams with benchmark test set"
        # Check phrase containment
        doc_lower = doc_text.lower()
        for phrase in self.benchmark_phrases:
            if len(phrase) > 30 and phrase in doc_lower:
                return True, f"Exact phrase match with benchmark test item: '{phrase[:40]}...'"
        return False, ""

def test_pipeline():
    tok = get_bpe_tokenizer()
    benchmarks = [
        "fresh_heldout_benchmark_50.json",
        "benchmark_v2_100.json",
        "data/benchmarks/benchmark_v3_sealed_300.json"
    ]
    auditor = BenchmarkContaminationAuditor(tok, benchmarks, n=8)
    sample_clean = "This is a clean encyclopedic text about planetary motion and gravitational physics."
    tokens = tok.encode(sample_clean)
    contam, reason = auditor.is_contaminated(tokens, sample_clean)
    print(f"Clean text test: contaminated={contam}, reason='{reason}'")
    assert not contam, "Clean text should not be flagged"
    print("Contamination filter initialization verified.")

if __name__ == "__main__":
    test_pipeline()
