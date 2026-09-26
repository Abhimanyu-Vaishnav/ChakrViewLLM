"""
ChakrView Indigenous Tokenizer Package.

Exposes the minimal Byte-Level BPE research prototype.
"""

from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
    byte_seq_to_token_ids,
    byte_to_token_id,
    is_byte_token,
    token_id_to_byte,
    token_ids_to_byte_seq,
)
from chakrview.tokenizer.bpe import (
    apply_merge,
    count_pairs,
    create_base_vocab,
    encode_with_merges,
    select_best_pair,
    train_toy_bpe,
)
from chakrview.tokenizer.decoder import decode_bytes, decode_tokens
from chakrview.tokenizer.encoder import encode_bytes, encode_text
from chakrview.tokenizer.special_tokens import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_ID_TO_TOKEN,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_TO_ID,
    is_special_token,
)
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.trainer import (
    BPETrainer,
    load_experiment_artifacts,
    save_experiment_artifacts,
)
from chakrview.tokenizer.interface import (
    D_MODEL,
    MAX_CONTEXT,
    PROVISIONAL_VOCAB_SIZE,
    calculate_static_parameter_memory,
    chunk_tokens,
    fake_embedding_lookup,
    prepare_batch,
    truncate_tokens,
    validate_token_ids,
)

__all__ = [
    # Special Tokens
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "SPECIAL_ID_TO_TOKEN",
    "SPECIAL_TOKEN_TO_ID",
    "SPECIAL_TOKEN_IDS",
    "NUM_SPECIAL_TOKENS",
    "is_special_token",
    # Byte Vocabulary
    "BYTE_OFFSET",
    "NUM_BYTE_TOKENS",
    "byte_to_token_id",
    "token_id_to_byte",
    "is_byte_token",
    "byte_seq_to_token_ids",
    "token_ids_to_byte_seq",
    # BPE Engine
    "create_base_vocab",
    "count_pairs",
    "select_best_pair",
    "apply_merge",
    "train_toy_bpe",
    "encode_with_merges",
    # Encoder & Decoder Functional API
    "encode_bytes",
    "encode_text",
    "decode_bytes",
    "decode_tokens",
    # Model Class & Trainer
    "BPETokenizer",
    "BPETrainer",
    "save_experiment_artifacts",
    "load_experiment_artifacts",
    # Interface Contract & Handoff
    "PROVISIONAL_VOCAB_SIZE",
    "MAX_CONTEXT",
    "D_MODEL",
    "validate_token_ids",
    "truncate_tokens",
    "chunk_tokens",
    "prepare_batch",
    "fake_embedding_lookup",
    "calculate_static_parameter_memory",
]


