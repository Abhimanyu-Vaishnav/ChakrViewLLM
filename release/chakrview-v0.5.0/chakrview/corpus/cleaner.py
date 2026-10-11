"""
ChakrView Corpus Pipeline: Document Cleaner Module.

Enforces clear separation across document stages:
A. RAW TEXT (unmodified bytes)
B. CLEANED TRAINING TEXT (safe whitespace regularization for merge counting)
C. TOKENIZER BENCHMARK TEXT (clean evaluation text)
D. VALIDATION / ADVERSARIAL TEXT (raw octets, zero-width chars, CRLF preserved)

Critical Invariant:
Cleaning NEVER applies destructive normalization that loses information or alters
Devanagari half-forms / ZWJ / ZWNJ characters.
"""

from typing import Literal

CorpusStage = Literal["raw", "training", "benchmark", "adversarial"]


def clean_training_text(text: str) -> str:
    """
    Safe cleaning for training merge learning.
    - Strips non-semantic trailing carriage returns.
    - Replaces isolated \\r with \\n.
    - Preserves all internal spaces, tabs, Unicode characters, and Devanagari ligatures.
    """
    # Replace CRLF with LF for consistent cross-platform line counting
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on the document line
    return cleaned.rstrip()


def clean_benchmark_text(text: str) -> str:
    """
    Preparation for benchmark evaluation text.
    Preserves exact text characters, stripping only terminal line breaks.
    """
    return text.rstrip("\r\n")


def preserve_adversarial_text(text: str) -> str:
    """
    Strict identity function for validation and adversarial test cases.
    Zero alterations, preserving CRLF, ZWJ, ZWNJ, tabs, and raw octets.
    """
    return text


def prepare_text_by_stage(text: str, stage: CorpusStage = "training") -> str:
    """
    Format text according to its designated corpus stage.
    """
    if stage == "training":
        return clean_training_text(text)
    elif stage == "benchmark":
        return clean_benchmark_text(text)
    elif stage in ("raw", "adversarial"):
        return preserve_adversarial_text(text)
    else:
        raise ValueError(f"Unknown corpus stage: {stage}")
