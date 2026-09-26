"""
ChakrView Tokenizer Corpus Pipeline: Statistics Module.

Analyzes and reports comprehensive quantitative metrics across the tokenizer corpus.
"""

import unicodedata
from typing import Any, Dict, List
from chakrview.tokenizer_corpus.dedup import deduplicate_lines
from chakrview.tokenizer_corpus.loader import CorpusItem


def identify_script(char: str) -> str:
    """Classify a Unicode character into primary script / block category."""
    cp = ord(char)
    name = unicodedata.name(char, "").upper()

    if 0x00 <= cp <= 0x7F:
        if char.isspace():
            return "Whitespace"
        if char.isnumeric():
            return "ASCII Digit"
        if char.isalpha():
            return "ASCII Latin"
        return "ASCII Punct/Symbol"

    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0980 <= cp <= 0x0D7F:
        return "Other Indic"
    if 0x2000 <= cp <= 0x206F:
        return "General Punctuation"
    if 0x2100 <= cp <= 0x214F or 0x2200 <= cp <= 0x22FF:
        return "Math & Letterlike"
    if 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
        return "Emoji & Pictograph"
    if "DEVANAGARI" in name:
        return "Devanagari"
    if "LATIN" in name:
        return "Latin Extended"
    return "Other Unicode"


def compute_corpus_statistics(corpus: Dict[str, List[CorpusItem]]) -> Dict[str, Any]:
    """
    Compute comprehensive metrics across the entire corpus and per category.
    """
    all_items: List[CorpusItem] = []
    for items in corpus.values():
        all_items.extend(items)

    total_docs = len(all_items)
    total_chars = sum(len(item.text) for item in all_items)
    total_bytes = sum(len(item.text.encode("utf-8")) for item in all_items)
    total_whitespace = sum(sum(1 for c in item.text if c.isspace()) for item in all_items)

    # Deduplication metrics
    _, duplicate_count = deduplicate_lines(all_items)

    # Script distribution
    script_counts: Dict[str, int] = {}
    for item in all_items:
        for c in item.text:
            s = identify_script(c)
            script_counts[s] = script_counts.get(s, 0) + 1

    # Longest and shortest samples
    if all_items:
        longest_item = max(all_items, key=lambda x: len(x.text))
        shortest_item = min(all_items, key=lambda x: len(x.text))
        longest_info = {"length": len(longest_item.text), "category": longest_item.category, "snippet": longest_item.text[:60]}
        shortest_info = {"length": len(shortest_item.text), "category": shortest_item.category, "snippet": shortest_item.text[:60]}
    else:
        longest_info = {"length": 0, "category": "", "snippet": ""}
        shortest_info = {"length": 0, "category": "", "snippet": ""}

    # Category breakdown
    category_stats: Dict[str, Dict[str, Any]] = {}
    for cat, items in sorted(corpus.items()):
        c_docs = len(items)
        c_chars = sum(len(x.text) for x in items)
        c_bytes = sum(len(x.text.encode("utf-8")) for x in items)
        c_ws = sum(sum(1 for c in x.text if c.isspace()) for x in items)
        c_longest = max(len(x.text) for x in items) if items else 0
        c_shortest = min(len(x.text) for x in items) if items else 0
        category_stats[cat] = {
            "documents": c_docs,
            "characters": c_chars,
            "bytes": c_bytes,
            "whitespace": c_ws,
            "bytes_pct": round((c_bytes / total_bytes * 100), 2) if total_bytes > 0 else 0.0,
            "longest_chars": c_longest,
            "shortest_chars": c_shortest,
        }

    return {
        "total_documents": total_docs,
        "total_characters": total_chars,
        "total_bytes": total_bytes,
        "total_whitespace": total_whitespace,
        "duplicate_count": duplicate_count,
        "unique_documents": total_docs - duplicate_count,
        "longest_sample": longest_info,
        "shortest_sample": shortest_info,
        "script_distribution": script_counts,
        "category_stats": category_stats,
    }


def format_statistics_report(stats: Dict[str, Any]) -> str:
    """Format the corpus statistics dictionary into a clean markdown string."""
    lines: List[str] = [
        "# ChakrView Tokenizer Corpus Statistics Report\n",
        f"- **Total Documents / Lines**: {stats['total_documents']:,}",
        f"- **Unique Documents**: {stats['unique_documents']:,}",
        f"- **Duplicate Lines**: {stats['duplicate_count']:,}",
        f"- **Total Characters**: {stats['total_characters']:,}",
        f"- **Total UTF-8 Bytes**: {stats['total_bytes']:,}",
        f"- **Total Whitespace Characters**: {stats['total_whitespace']:,}",
        f"- **Longest Sample**: {stats['longest_sample']['length']} chars ({stats['longest_sample']['category']}: \"{stats['longest_sample']['snippet']}...\")",
        f"- **Shortest Sample**: {stats['shortest_sample']['length']} chars ({stats['shortest_sample']['category']}: \"{stats['shortest_sample']['snippet']}\")\n",
        "## Category Distribution Breakdown\n",
        "| Category | Documents | Characters | UTF-8 Bytes | Byte Share (%) | Whitespace | Max Len | Min Len |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat, c in stats["category_stats"].items():
        lines.append(
            f"| `{cat}` | {c['documents']} | {c['characters']:,} | {c['bytes']:,} | {c['bytes_pct']}% | {c['whitespace']:,} | {c['longest_chars']} | {c['shortest_chars']} |"
        )

    lines.append("\n## Script & Unicode Character Distribution\n")
    lines.append("| Character Category / Script | Count | Share (%) |")
    lines.append("| :--- | :---: | :---: |")

    total_chars = stats["total_characters"]
    for script, count in sorted(stats["script_distribution"].items(), key=lambda x: -x[1]):
        pct = round(count / total_chars * 100, 2) if total_chars > 0 else 0.0
        lines.append(f"| {script} | {count:,} | {pct}% |")

    return "\n".join(lines)
