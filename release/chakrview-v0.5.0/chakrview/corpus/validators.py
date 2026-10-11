"""
ChakrView Corpus Pipeline: Data Quality Validator Module.

Strictly checks all corpus items against data quality criteria:
- empty documents
- duplicate documents
- extremely short samples (< min_chars)
- extremely long samples (> max_chars)
- malformed UTF-8 / surrogate codepoints
- unexpected control characters
- invalid Unicode sequences
- excessive repeated characters
- pathological whitespace
- accidental binary files / null bytes

Produces a transparent, machine-readable validation report with explicit
reasons for any discarded or flagged documents.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.corpus.loader import CorpusDocument

# Control character pattern: ASCII 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F
CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SURROGATE_REGEX = re.compile(r"[\ud800-\udfff]")


@dataclass(frozen=True)
class ValidationIssue:
    """Detailed record of a data quality violation."""
    line_index: int
    source_file: str
    category: str
    issue_type: str
    reason: str
    snippet: str
    action: str  # 'DISCARD' or 'FLAG'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_single_text(
    text: str,
    source_file: str = "unknown",
    category: str = "unknown",
    line_index: int = 1,
    min_chars: int = 2,
    max_chars: int = 10000,
    max_repeated_chars: int = 15,
) -> List[ValidationIssue]:
    """
    Validate a single text string against data quality rules.
    """
    issues: List[ValidationIssue] = []

    # 1. Empty / Whitespace-only check
    if not text or not text.strip():
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="EMPTY_DOCUMENT",
                reason="Document is empty or contains only whitespace",
                snippet="",
                action="DISCARD",
            )
        )
        return issues

    # 2. Null byte / Accidental binary check
    if "\x00" in text:
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="BINARY_NULL_BYTE",
                reason="Document contains null byte (0x00); potential accidental binary",
                snippet=repr(text[:30]),
                action="DISCARD",
            )
        )

    # 3. Unexpected control characters
    ctrl_matches = CONTROL_CHAR_REGEX.findall(text)
    if ctrl_matches:
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="UNEXPECTED_CONTROL_CHARS",
                reason=f"Document contains {len(ctrl_matches)} illegal control characters",
                snippet=repr(text[:40]),
                action="DISCARD",
            )
        )

    # 4. Surrogate codepoints
    if SURROGATE_REGEX.search(text):
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="SURROGATE_CODEPOINT",
                reason="Document contains lone surrogate codepoints (U+D800..U+DFFF)",
                snippet=repr(text[:40]),
                action="DISCARD",
            )
        )

    # 5. Length bounds
    c_len = len(text)
    if c_len < min_chars:
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="EXTREMELY_SHORT",
                reason=f"Document length ({c_len} chars) is below threshold of {min_chars}",
                snippet=text,
                action="DISCARD",
            )
        )
    elif c_len > max_chars:
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="EXTREMELY_LONG",
                reason=f"Document length ({c_len} chars) exceeds safety threshold of {max_chars}",
                snippet=text[:50] + "...",
                action="FLAG",
            )
        )

    # 6. Excessive repeated characters
    repeat_match = re.search(rf"(.)\1{{{max_repeated_chars},}}", text)
    if repeat_match:
        rep_char = repeat_match.group(1)
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="EXCESSIVE_REPEATED_CHARS",
                reason=f"Repeated character {repr(rep_char)} appears >{max_repeated_chars} consecutive times",
                snippet=repeat_match.group(0)[:30] + "...",
                action="FLAG",
            )
        )

    # 7. Pathological whitespace (e.g. >10 consecutive tabs or newlines)
    if re.search(r"[\t]{10,}|\r{5,}", text):
        issues.append(
            ValidationIssue(
                line_index=line_index,
                source_file=source_file,
                category=category,
                issue_type="PATHOLOGICAL_WHITESPACE",
                reason="Excessive consecutive whitespace/carriage returns detected",
                snippet=repr(text[:40]),
                action="FLAG",
            )
        )

    return issues


def validate_corpus_collection(
    documents: List[CorpusDocument],
    output_report_path: Optional[Path] = None,
) -> Tuple[List[CorpusDocument], Dict[str, Any]]:
    """
    Validate a full collection of documents.
    Identifies duplicates and invalid documents without silent data loss.

    Returns:
        (valid_documents, report_summary)
    """
    seen_texts: Set[str] = set()
    all_issues: List[ValidationIssue] = []
    valid_docs: List[CorpusDocument] = []
    duplicate_count = 0
    discard_count = 0

    for idx, doc in enumerate(documents, start=1):
        # 1. Duplicate check
        norm_key = doc.text.strip()
        if norm_key in seen_texts:
            duplicate_count += 1
            all_issues.append(
                ValidationIssue(
                    line_index=doc.line_number,
                    source_file=doc.source_file,
                    category=doc.category,
                    issue_type="EXACT_DUPLICATE",
                    reason="Exact duplicate of an earlier line in the corpus",
                    snippet=doc.text[:50],
                    action="DISCARD",
                )
            )
            continue
        seen_texts.add(norm_key)

        # 2. Quality rules
        doc_issues = validate_single_text(
            text=doc.text,
            source_file=doc.source_file,
            category=doc.category,
            line_index=doc.line_number,
        )

        has_discard = any(issue.action == "DISCARD" for issue in doc_issues)
        all_issues.extend(doc_issues)

        if has_discard:
            discard_count += 1
        else:
            valid_docs.append(doc)

    report_summary: Dict[str, Any] = {
        "total_documents_checked": len(documents),
        "valid_documents": len(valid_docs),
        "duplicate_documents": duplicate_count,
        "discarded_documents": discard_count,
        "total_issues_found": len(all_issues),
        "issue_breakdown": {},
        "issues": [issue.to_dict() for issue in all_issues],
    }

    # Aggregate issue types
    for issue in all_issues:
        report_summary["issue_breakdown"][issue.issue_type] = (
            report_summary["issue_breakdown"].get(issue.issue_type, 0) + 1
        )

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return valid_docs, report_summary
