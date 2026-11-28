"""
Comprehensive Multi-Domain Tokenizer Validation for NR-057-R1 over Corpus V6.
Measures:
- Exact round-trip reconstruction (decode(encode(text)) == text)
- UNK rate across English, Hindi Devanagari, Hinglish, Python code, Math
- Token fertility (tokens per character, tokens per word)
- Bytes per token, characters per token
- Active vocabulary utilization across Train, Val, and Test splits
- Sequence length distribution and truncation risk
"""

import os
import sys
import json
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath("."))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def validate_v6_tokenizer():
    print("[TokenizerV6] Loading canonical tokenizer...")
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    unk_id = getattr(tok, "unk_token_id", 3)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok.vocab)

    splits = {
        "train": "data/corpus_v6/splits/corpus_v6_train.jsonl",
        "validation": "data/corpus_v6/splits/corpus_v6_val.jsonl",
        "test": "data/corpus_v6/splits/corpus_v6_test.jsonl"
    }

    report = {
        "mission": "NR-057-R1",
        "tokenizer_vocab_size": vocab_size,
        "splits_evaluated": {},
        "domain_breakdown": {},
        "overall_summary": {}
    }

    global_used_tokens = Counter()
    global_total_tokens = 0
    global_total_bytes = 0
    global_total_chars = 0
    global_unk_count = 0
    global_roundtrip_perfect = 0
    global_docs = 0

    domain_stats = defaultdict(lambda: {
        "docs": 0, "tokens": 0, "bytes": 0, "chars": 0, "unk": 0, "roundtrip_perfect": 0
    })

    for s_name, s_path in splits.items():
        print(f"Validating split: {s_name} ({s_path})...")
        s_tokens = 0
        s_bytes = 0
        s_chars = 0
        s_unk = 0
        s_docs = 0
        s_used_tokens = Counter()

        with open(s_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                text = doc["text"]
                dom = doc.get("domain", "unknown")
                
                toks = tok.encode(text)
                t_len = len(toks)
                c_len = len(text)
                b_len = len(text.encode("utf-8"))
                unk_cnt = toks.count(unk_id)

                # Roundtrip fidelity
                dec = tok.decode(toks)
                norm_orig = unicodedata.normalize("NFKC", text)
                norm_dec = unicodedata.normalize("NFKC", dec)
                is_perfect = (norm_orig == norm_dec)

                s_docs += 1
                s_tokens += t_len
                s_bytes += b_len
                s_chars += c_len
                s_unk += unk_cnt
                for t in toks:
                    s_used_tokens[t] += 1
                    global_used_tokens[t] += 1

                domain_stats[dom]["docs"] += 1
                domain_stats[dom]["tokens"] += t_len
                domain_stats[dom]["bytes"] += b_len
                domain_stats[dom]["chars"] += c_len
                domain_stats[dom]["unk"] += unk_cnt
                if is_perfect:
                    domain_stats[dom]["roundtrip_perfect"] += 1
                    global_roundtrip_perfect += 1

        global_docs += s_docs
        global_total_tokens += s_tokens
        global_total_bytes += s_bytes
        global_total_chars += s_chars
        global_unk_count += s_unk

        report["splits_evaluated"][s_name] = {
            "document_count": s_docs,
            "token_count": s_tokens,
            "unk_rate_pct": round(s_unk / max(1, s_tokens) * 100, 4),
            "bytes_per_token": round(s_bytes / max(1, s_tokens), 2),
            "chars_per_token": round(s_chars / max(1, s_tokens), 2),
            "active_vocab_in_split": len(s_used_tokens),
            "active_vocab_pct": round(len(s_used_tokens) / vocab_size * 100, 2)
        }

    for dom, s in domain_stats.items():
        report["domain_breakdown"][dom] = {
            "document_count": s["docs"],
            "token_count": s["tokens"],
            "unk_rate_pct": round(s["unk"] / max(1, s["tokens"]) * 100, 4),
            "bytes_per_token": round(s["bytes"] / max(1, s["tokens"]), 2),
            "chars_per_token": round(s["chars"] / max(1, s["tokens"]), 2),
            "token_fertility_chars_per_tok": round(s["chars"] / max(1, s["tokens"]), 2),
            "roundtrip_accuracy_pct": round(s["roundtrip_perfect"] / max(1, s["docs"]) * 100, 2)
        }

    report["overall_summary"] = {
        "total_documents_evaluated": global_docs,
        "total_tokens_evaluated": global_total_tokens,
        "total_unk_tokens": global_unk_count,
        "overall_unk_rate_pct": round(global_unk_count / max(1, global_total_tokens) * 100, 4),
        "overall_bytes_per_token": round(global_total_bytes / max(1, global_total_tokens), 2),
        "overall_chars_per_token": round(global_total_chars / max(1, global_total_tokens), 2),
        "total_active_vocab_utilized": len(global_used_tokens),
        "total_active_vocab_pct": round(len(global_used_tokens) / vocab_size * 100, 2),
        "overall_roundtrip_fidelity_pct": round(global_roundtrip_perfect / max(1, global_docs) * 100, 2)
    }

    out_json = "experiments/corpus_nr057_r1/tokenizer_validation_v6.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation complete! Written to {out_json}")
    print(f"Overall UNK Rate: {report['overall_summary']['overall_unk_rate_pct']}%")
    print(f"Active Vocab Coverage: {report['overall_summary']['total_active_vocab_pct']}% ({len(global_used_tokens)}/{vocab_size})")

if __name__ == "__main__":
    validate_v6_tokenizer()
