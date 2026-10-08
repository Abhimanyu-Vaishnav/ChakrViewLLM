"""
ChakrView Tokenizer: BPETokenizer Model Class.

Provides the unified, high-level interface for the Byte-Level BPE research prototype.
"""

from typing import Any, Dict, List, Optional, Tuple

from chakrview.tokenizer.bpe import create_base_vocab, train_toy_bpe
from chakrview.tokenizer.decoder import decode_bytes, decode_tokens
from chakrview.tokenizer.encoder import encode_bytes, encode_text
from chakrview.tokenizer.special_tokens import (
    BOS_ID,
    EOS_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    SPECIAL_ID_TO_TOKEN,
    SPECIAL_TOKEN_TO_ID,
)


class BPETokenizer:
    """
    Indigenous Byte-Level BPE Tokenizer Prototype for ChakrView.

    Architectural Invariant:
      Decode(Encode(text)) == text for all supported inputs.
      Correctness strictly precedes compression.
    """

    def __init__(
        self,
        merges: Optional[Dict[Tuple[int, int], int]] = None,
        vocab: Optional[Dict[int, bytes]] = None,
    ) -> None:
        self._merges: Dict[Tuple[int, int], int] = dict(merges) if merges else {}
        self._vocab: Dict[int, bytes] = dict(vocab) if vocab else create_base_vocab()

    @property
    def merges(self) -> Dict[Tuple[int, int], int]:
        """Return the dictionary of learned merges (pair -> new_token_id)."""
        return self._merges

    @property
    def vocab(self) -> Dict[int, bytes]:
        """Return the token ID to byte mapping."""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens, base bytes, and merges."""
        return NUM_SPECIAL_TOKENS + len(self._vocab)

    @property
    def num_merges(self) -> int:
        """Total number of learned merges in this tokenizer."""
        return len(self._merges)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode Unicode text into a deterministic list of token IDs.
        """
        return encode_text(
            text=text,
            merges=self._merges,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        """
        Decode a sequence of token IDs back into exact Unicode text.
        """
        return decode_tokens(
            tokens=tokens,
            vocab=self._vocab,
            skip_special_tokens=skip_special_tokens,
            errors=errors,
        )

    def encode_bytes(self, data: bytes) -> List[int]:
        """
        Encode an arbitrary sequence of raw 8-bit bytes.
        """
        return encode_bytes(data=data, merges=self._merges)

    def decode_bytes(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True,
    ) -> bytes:
        """
        Decode a sequence of token IDs back into raw 8-bit bytes.
        """
        return decode_bytes(
            tokens=tokens,
            vocab=self._vocab,
            skip_special_tokens=skip_special_tokens,
        )

    def train_toy(
        self,
        data: str | bytes,
        max_merges: int = 10,
        min_frequency: int = 2,
    ) -> None:
        """
        Train toy merges on a small sample corpus.
        Updates internal merges and vocabulary.
        """
        raw_bytes = data.encode("utf-8") if isinstance(data, str) else data
        merges, vocab = train_toy_bpe(
            data=raw_bytes,
            max_merges=max_merges,
            min_frequency=min_frequency,
        )
        self._merges.update(merges)
        self._vocab.update(vocab)
