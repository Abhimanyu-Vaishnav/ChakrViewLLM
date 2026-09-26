"""
ChakrView Tokenizer: Special Tokens Module.

Defines the minimalist special-token set for the ChakrView causal language model.
Complies with Step 2 ADR-T06:
  <BOS> = 0 (Beginning of Sequence)
  <EOS> = 1 (End of Sequence)
  <PAD> = 2 (Padding Token)

Total special tokens: exactly 3.
Tokens <UNK>, <MASK>, <SEP>, <NL> are strictly excluded.
"""

from typing import Dict, Final, Set

BOS_ID: Final[int] = 0
EOS_ID: Final[int] = 1
PAD_ID: Final[int] = 2

BOS_TOKEN: Final[str] = "<BOS>"
EOS_TOKEN: Final[str] = "<EOS>"
PAD_TOKEN: Final[str] = "<PAD>"

SPECIAL_ID_TO_TOKEN: Final[Dict[int, str]] = {
    BOS_ID: BOS_TOKEN,
    EOS_ID: EOS_TOKEN,
    PAD_ID: PAD_TOKEN,
}

SPECIAL_TOKEN_TO_ID: Final[Dict[str, int]] = {
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
    PAD_TOKEN: PAD_ID,
}

SPECIAL_TOKEN_IDS: Final[Set[int]] = {BOS_ID, EOS_ID, PAD_ID}
NUM_SPECIAL_TOKENS: Final[int] = len(SPECIAL_TOKEN_IDS)


def is_special_token(token_id: int) -> bool:
    """Return True if the token ID is a designated special token."""
    return token_id in SPECIAL_TOKEN_IDS
