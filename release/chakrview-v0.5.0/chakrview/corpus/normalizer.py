"""
ChakrView Corpus Pipeline: Unicode Normalization Policy Module.

Explicitly formalizes the separation between training text preprocessing
and runtime tokenizer inference.

Architectural Invariant:
RUNTIME ENCODING IS STRICTLY LOSSLESS.
No Unicode normalization (NFC, NFD, NFKC, NFKD) is ever applied at runtime,
as modifying Unicode forms violates Decode(Encode(S)) == S for distinct byte representations.
"""

import unicodedata
from typing import Dict

NORMALIZATION_POLICIES: Dict[str, Dict[str, str]] = {
    "training": {
        "policy": "NFC or IDENTITY",
        "description": "Standardizes composed characters during merge extraction if configured",
    },
    "runtime": {
        "policy": "STRICT IDENTITY (NO NORMALIZATION)",
        "description": "Preserves original octet sequences exactly to maintain lossless reconstruction",
    },
}


def normalize_training_text(text: str, form: str = "NFC") -> str:
    """
    Optional composition normalization for training text only.
    """
    if form == "IDENTITY":
        return text
    return unicodedata.normalize(form, text)


def runtime_identity(text: str) -> str:
    """
    Runtime identity pass-through. Preserves all codepoints and byte sequences.
    """
    return text
