r"""
ChakrView Numeric Tokenization Adapters (Phase 3.6).

Evaluates three candidate numeric tokenization strategies:
- Candidate A: Single-digit tokenization (splits digit sequences into individual digits \d)
- Candidate B: Two-digit chunks (splits digit sequences into common 2-digit chunks \d{1,2})
- Candidate C: Normal BPE frequency learning (unconstrained BPE on raw bytes)

Ensures exact round-trip reconstruction invariant:
Decode(Encode(x)) == x
"""

import re
from typing import List, Tuple

from chakrview.tokenizer.tokenizer import BPETokenizer


class NumericTokenizationAdapter:
    """
    Adapter providing Candidate A, B, and C numeric tokenization strategies
    wrapped around a BPETokenizer instance.
    """

    def __init__(self, tokenizer: BPETokenizer, strategy: str = "C") -> None:
        if strategy not in ("A", "B", "C"):
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'A', 'B', or 'C'.")
        self.tokenizer = tokenizer
        self.strategy = strategy

    def encode(self, text: str) -> List[int]:
        """
        Encode text using the configured numeric strategy.
        """
        if self.strategy == "C":
            # Candidate C: Normal BPE frequency learning
            return self.tokenizer.encode(text)

        # For Candidate A and B, pre-segment digits using regex
        if self.strategy == "A":
            # Match individual digits or non-digit chunks
            pattern = r"(\d|\D+)"
        elif self.strategy == "B":
            # Match 2-digit chunks, 1-digit, or non-digit chunks
            pattern = r"(\d{1,2}|\D+)"
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        chunks = [m for m in re.findall(pattern, text) if m]
        tokens: List[int] = []
        for chunk in chunks:
            tokens.extend(self.tokenizer.encode(chunk))
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode tokens back to text.
        Decoding is invariant across numeric strategies because subwords/bytes
        concatenate to the exact original byte stream.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def round_trip_check(self, text: str) -> Tuple[bool, List[int], str]:
        """
        Verify lossless round-trip reconstruction for given text:
        Decode(Encode(text)) == text.
        """
        tokens = self.encode(text)
        decoded = self.decode(tokens)
        return (decoded == text), tokens, decoded
