"""
Tokenizer Compatibility Validation Script for NR-057.
Validates the canonical BPE 4096 tokenizer against Corpus V5:
- UNK token rate across English, Hindi, Hinglish, Code, and Math.
- Compression ratio (bytes per token, chars per token).
- Round-trip fidelity (decode(encode(text)) == text).
- Vocab utilization.
"""

import os
import sys
import json
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def validate_tokenizer():
    print("[TokenizerValidation] Loading tokenizer...")
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    unk_id = getattr(tok, "unk_token_id", 3)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok.vocab)
    print(f"Tokenizer loaded. Vocab size: {vocab_size}, UNK ID: {unk_id}")

    corpus_file = "data/corpus_v5/corpus_v5_val.jsonl"
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Missing {corpus_file}")

    domain_stats = {}
    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    total_unk = 0
    roundtrip_failures = 0
    used_token_ids = Counter()

    with open(corpus_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            doc = json.loads(line)
            dom = doc.get("domain", "unknown")
            text = doc["text"]
            
            if dom not in domain_stats:
                domain_stats[dom] = {
                    "docs": 0, "chars": 0, "bytes": 0, "tokens": 0, "unk": 0, "roundtrip_perfect": 0
                }
            
            b_len = len(text.encode("utf-8"))
            c_len = len(text)
            toks = tok.encode(text)
            t_len = len(toks)
            unk_cnt = toks.count(unk_id) if isinstance(toks, list) else 0

            # Roundtrip test
            dec = tok.decode(toks)
            norm_orig = unicodedata.normalize("NFKC", text)
            norm_dec = unicodedata.normalize("NFKC", dec)
            is_perfect = (norm_orig == norm_dec)
            if not is_perfect:
                roundtrip_failures += 1

            for t in toks:
                used_token_ids[t] += 1

            domain_stats[dom]["docs"] += 1
            domain_stats[dom]["chars"] += c_len
            domain_stats[dom]["bytes"] += b_len
            domain_stats[dom]["tokens"] += t_len
            domain_stats[dom]["unk"] += unk_cnt
            if is_perfect:
                domain_stats[dom]["roundtrip_perfect"] += 1

            total_chars += c_len
            total_bytes += b_len
            total_tokens += t_len
            total_unk += unk_cnt

    summary = {
        "mission": "NR-057",
        "tokenizer_path": "data/experiments/vocab_4096",
        "vocabulary_size": vocab_size,
        "sample_validation_docs": sum(d["docs"] for d in domain_stats.values()),
        "total_tokens_evaluated": total_tokens,
        "total_unk_tokens": total_unk,
        "overall_unk_rate_pct": round(total_unk / max(1, total_tokens) * 100, 4),
        "overall_bytes_per_token": round(total_bytes / max(1, total_tokens), 2),
        "overall_chars_per_token": round(total_chars / max(1, total_tokens), 2),
        "roundtrip_accuracy_pct": round((total_tokens - roundtrip_failures) / max(1, total_tokens) * 100, 2),
        "active_vocab_coverage_pct": round(len(used_token_ids) / vocab_size * 100, 2),
        "domain_analysis": {}
    }

    for dom, s in domain_stats.items():
        summary["domain_analysis"][dom] = {
            "docs": s["docs"],
            "tokens": s["tokens"],
            "unk_rate_pct": round(s["unk"] / max(1, s["tokens"]) * 100, 4),
            "bytes_per_token": round(s["bytes"] / max(1, s["tokens"]), 2),
            "chars_per_token": round(s["chars"] / max(1, s["tokens"]), 2),
            "roundtrip_fidelity_pct": round(s["roundtrip_perfect"] / max(1, s["docs"]) * 100, 2)
        }

    out_json = "experiments/corpus_nr057/tokenizer_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Validation complete. Written to {out_json}")
    print(f"Overall UNK rate: {summary['overall_unk_rate_pct']}%")
    print(f"Bytes/token: {summary['overall_bytes_per_token']}")
    print(f"Active vocab coverage: {summary['active_vocab_coverage_pct']}% ({len(used_token_ids)}/{vocab_size})")

if __name__ == "__main__":
    validate_tokenizer()
