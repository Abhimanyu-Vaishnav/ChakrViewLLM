"""
ChakrView Tokenizer: Neural-Core Interface Contract Module.

Codifies the runtime contracts, data shapes, bounds validation, batch preparation,
and memory accounting between the tokenizer and the neural core.

Architecture Status:
-------------------
VOCAB_SIZE = 4096 (PROVISIONAL / EXPERIMENTALLY SELECTED from Step 2.3 benchmark).
MAX_CONTEXT = 512 (Target context window for Chakr-Micro).
D_MODEL = 192 (Hidden embedding dimension for Chakr-Micro).
WEIGHT_TYING = TRUE (Output projection W_out = E^T).
"""

from typing import Any, Dict, Final, List, Optional, Tuple

from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID

# Provisional Architecture Constants
PROVISIONAL_VOCAB_SIZE: Final[int] = 4096
MAX_CONTEXT: Final[int] = 512
D_MODEL: Final[int] = 192

# Token ID Layout Constants
SPECIAL_TOKEN_START: Final[int] = 0
SPECIAL_TOKEN_END: Final[int] = 2  # BOS=0, EOS=1, PAD=2
BYTE_TOKEN_START: Final[int] = 3
BYTE_TOKEN_END: Final[int] = 258   # 256 fundamental bytes
MERGE_TOKEN_START: Final[int] = 259


def validate_token_ids(
    tokens: List[int],
    vocab_size: int = PROVISIONAL_VOCAB_SIZE,
) -> None:
    """
    Validate that all token IDs in the sequence fall strictly within [0, vocab_size - 1].
    Raises ValueError on out-of-bounds token IDs.
    """
    if not isinstance(tokens, list):
        raise TypeError(f"Expected list of int, got {type(tokens).__name__}")

    for idx, tid in enumerate(tokens):
        if not isinstance(tid, int):
            raise TypeError(f"Token at index {idx} is not an integer: {tid!r}")
        if not (0 <= tid < vocab_size):
            raise ValueError(
                f"Token ID {tid} at index {idx} violates valid vocabulary range [0, {vocab_size - 1}]"
            )


def truncate_tokens(
    tokens: List[int],
    max_length: int = MAX_CONTEXT,
    add_eos_if_truncated: bool = False,
) -> List[int]:
    """
    Explicit runtime truncation policy for sequences exceeding neural context length.

    Notice: The tokenizer does not silently truncate text. Truncation must be explicitly
    requested via runtime policy.
    """
    if len(tokens) <= max_length:
        return list(tokens)

    if add_eos_if_truncated:
        truncated = tokens[: max_length - 1]
        truncated.append(EOS_ID)
        return truncated

    return list(tokens[:max_length])


def chunk_tokens(
    tokens: List[int],
    chunk_size: int = MAX_CONTEXT,
    overlap: int = 0,
) -> List[List[int]]:
    """
    Partition a long token sequence into fixed-size chunks with optional sliding-window overlap.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not (0 <= overlap < chunk_size):
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")

    if not tokens:
        return [[]]

    step = chunk_size - overlap
    chunks: List[List[int]] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        if len(chunk) < chunk_size:
            break
        i += step

    return chunks


def prepare_batch(
    token_sequences: List[List[int]],
    pad_id: int = PAD_ID,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Prepare batched input tensors and attention masks for the neural core.

    Args:
        token_sequences: List of variable-length token ID lists.
        pad_id: Padding token ID (default: PAD_ID = 2).
        max_seq_len: Optional fixed sequence length. Defaults to max length in batch.

    Returns:
        input_ids: 2D list of shape [batch_size, sequence_length].
        attention_mask: 2D list of shape [batch_size, sequence_length],
                        where 1 indicates valid token and 0 indicates padding.
    """
    if not token_sequences:
        return [], []

    batch_max = max(len(seq) for seq in token_sequences)
    target_len = max_seq_len if max_seq_len is not None else batch_max

    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []

    for seq in token_sequences:
        seq_len = len(seq)
        if seq_len > target_len:
            # Explicit truncation if longer than target_len
            padded_seq = seq[:target_len]
            mask = [1] * target_len
        else:
            padding_len = target_len - seq_len
            padded_seq = seq + [pad_id] * padding_len
            mask = [1] * seq_len + [0] * padding_len

        input_ids.append(padded_seq)
        attention_mask.append(mask)

    return input_ids, attention_mask


def fake_embedding_lookup(
    token_ids: List[int],
    embedding_weights: Optional[List[List[float]]] = None,
    vocab_size: int = PROVISIONAL_VOCAB_SIZE,
    d_model: int = D_MODEL,
) -> List[List[float]]:
    """
    Simulate the neural-core embedding handoff:
        Token IDs (1D: [sequence_length]) -> Embedding Vectors (2D: [sequence_length, d_model])

    Proves that the token stream produced by the tokenizer can be consumed by the neural
    embedding layer without ambiguity.
    """
    validate_token_ids(token_ids, vocab_size=vocab_size)

    seq_len = len(token_ids)
    if seq_len == 0:
        return []

    # If mock weights are not supplied, generate deterministic synthetic weights
    if embedding_weights is None:
        # Vector for token_id: [tid * 0.001 + d * 0.0001 for d in range(d_model)]
        return [
            [(tid * 0.001) + (dim * 0.0001) for dim in range(d_model)]
            for tid in token_ids
        ]

    if len(embedding_weights) != vocab_size:
        raise ValueError(
            f"Embedding table must have {vocab_size} rows, got {len(embedding_weights)}"
        )

    return [embedding_weights[tid] for tid in token_ids]


def calculate_static_parameter_memory(
    vocab_size: int = PROVISIONAL_VOCAB_SIZE,
    d_model: int = D_MODEL,
) -> Dict[str, Any]:
    """
    Compute parameter count and static weight memory footprint for the embedding table.

    CRITICAL NOTE:
    These figures represent STATIC PARAMETER STORAGE only.
    They explicitly EXCLUDE:
    - Activations
    - Temporary tensors
    - KV cache
    - Memory allocator overhead
    - Runtime framework buffers
    """
    param_count = vocab_size * d_model  # 4096 * 192 = 786,432

    # Weight tying: output projection W_out = E^T adds 0 additional unique parameters
    bytes_fp32 = param_count * 4
    bytes_fp16 = param_count * 2
    bytes_int8 = param_count * 1
    bytes_int4 = int(param_count * 0.5)

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "embedding_parameters": param_count,
        "tied_unembedding_parameters": 0,  # Shared weights
        "total_embedding_parameters": param_count,
        "memory_fp32_bytes": bytes_fp32,
        "memory_fp32_mb": round(bytes_fp32 / (1024 * 1024), 3),
        "memory_fp16_bytes": bytes_fp16,
        "memory_fp16_mb": round(bytes_fp16 / (1024 * 1024), 3),
        "memory_int8_bytes": bytes_int8,
        "memory_int8_mb": round(bytes_int8 / (1024 * 1024), 3),
        "memory_int4_bytes": bytes_int4,
        "memory_int4_mb": round(bytes_int4 / (1024 * 1024), 3),
        "scope": "STATIC_PARAMETER_STORAGE_ONLY",
        "exclusions": [
            "activations",
            "runtime_allocator_overhead",
            "temporary_tensors",
            "kv_cache",
            "framework_overhead",
        ],
    }
