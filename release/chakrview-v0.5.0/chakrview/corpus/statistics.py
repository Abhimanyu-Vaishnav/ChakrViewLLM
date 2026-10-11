"""
ChakrView Corpus Pipeline: Detailed Statistical Profiler Module.

Computes comprehensive corpus metrics per Phase 3:
- document count, character count, byte count
- Unicode code-point statistics (min, max, unique count)
- script distribution (Devanagari, Latin, Digits, Punctuation, Math/Symbols, Other)
- whitespace distribution (spaces, tabs, LF, CRLF)
- digit frequency
- punctuation frequency
- duplicate rate
- language / category breakdown
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.corpus.loader import CorpusDocument


def classify_char(c: str) -> str:
    """Classify a single Unicode character into linguistic/script buckets."""
    cp = ord(c)
    cat = unicodedata.category(c)

    if cat.startswith("P") or c in "।॥":
        return "Punctuation"
    elif 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    elif ("a" <= c <= "z") or ("A" <= c <= "Z"):
        return "Latin_ASCII"
    elif "0" <= c <= "9":
        return "ASCII_Digit"
    elif cat.startswith("P"):
        return "Punctuation"
    elif cat.startswith("S"):
        return "Symbol_Math"
    elif c in " \t\n\r" or cat.startswith("Z"):
        return "Whitespace"
    elif 0x1F300 <= cp <= 0x1F9FF or 0x2600 <= cp <= 0x26FF:
        return "Emoji"
    elif cat.startswith("M"):
        return "Combining_Mark"
    elif cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
        return "Zero_Width"
    else:
        return "Other_Unicode"


def compute_corpus_statistics(
    documents: List[CorpusDocument],
    duplicate_count: int = 0,
) -> Dict[str, Any]:
    """
    Compute comprehensive character, script, codepoint, and category statistics.
    """
    total_docs = len(documents)
    total_chars = 0
    total_bytes = 0

    unique_codepoints: Set[int] = set()
    script_counter: Counter = Counter()
    digit_counter: Counter = Counter()
    punct_counter: Counter = Counter()
    whitespace_counter: Counter = Counter()
    category_doc_counter: Counter = Counter()
    category_byte_counter: Counter = Counter()

    for doc in documents:
        category_doc_counter[doc.category] += 1
        category_byte_counter[doc.category] += doc.byte_count
        total_chars += doc.char_count
        total_bytes += doc.byte_count

        text = doc.text
        # Check CRLF vs LF
        whitespace_counter["crlf"] += text.count("\r\n")
        whitespace_counter["lf"] += text.count("\n")
        whitespace_counter["tabs"] += text.count("\t")
        whitespace_counter["spaces"] += text.count(" ")

        for c in text:
            cp = ord(c)
            unique_codepoints.add(cp)
            s_class = classify_char(c)
            script_counter[s_class] += 1

            if "0" <= c <= "9":
                digit_counter[c] += 1
            elif unicodedata.category(c).startswith("P"):
                punct_counter[c] += 1

    total_script_chars = sum(script_counter.values()) or 1
    script_percentages = {
        k: round((v / total_script_chars) * 100, 2)
        for k, v in script_counter.most_common()
    }

    category_percentages = {
        k: round((v / total_bytes) * 100, 2) if total_bytes > 0 else 0.0
        for k, v in category_byte_counter.items()
    }

    dup_rate = round(duplicate_count / (total_docs + duplicate_count) * 100, 2) if (total_docs + duplicate_count) > 0 else 0.0

    return {
        "summary": {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "total_bytes": total_bytes,
            "bytes_per_char": round(total_bytes / total_chars, 4) if total_chars > 0 else 0.0,
            "duplicate_rate_percentage": dup_rate,
        },
        "unicode_codepoint_statistics": {
            "unique_codepoints_count": len(unique_codepoints),
            "min_codepoint": min(unique_codepoints) if unique_codepoints else 0,
            "max_codepoint": max(unique_codepoints) if unique_codepoints else 0,
            "min_codepoint_hex": hex(min(unique_codepoints)) if unique_codepoints else "0x0",
            "max_codepoint_hex": hex(max(unique_codepoints)) if unique_codepoints else "0x0",
        },
        "script_distribution": {
            "counts": dict(script_counter.most_common()),
            "percentages": script_percentages,
        },
        "whitespace_distribution": dict(whitespace_counter),
        "digit_frequency": dict(digit_counter.most_common()),
        "top_punctuation_frequency": dict(punct_counter.most_common(20)),
        "category_distribution": {
            "document_counts": dict(category_doc_counter),
            "byte_counts": dict(category_byte_counter),
            "byte_percentages": category_percentages,
        },
    }


def save_statistics_report(
    stats: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save statistics report to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_p
