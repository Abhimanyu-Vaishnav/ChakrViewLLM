"""
ChakrView Canonical Model Weight Hashing Specification.

Establishes the ONE authoritative, immutable weight hashing protocol for ChakrView:
1. Parameter ordering: Lexicographically sorted parameter names: sorted(model.named_parameters(), key=lambda x: x[0])
2. Parameter name encoding: UTF-8 encoded string bytes
3. Parameter byte serialization: Contiguous NumPy CPU array buffer bytes: param.detach().cpu().numpy().tobytes()
4. Dtype: Float32 (native model tensor dtype)
5. Endianness: Native IEEE 754 (Little-endian on standard x86_64 / ARM64)
6. Tensor shape handling: Invariant across canonical model architecture
7. Concatenation order: Interleaved (name_bytes + data_bytes) per parameter tensor in sorted order
8. Hash digest: SHA-256 hex digest

Canonical Baseline Invariants:
- Total Parameters: 3,443,136
- SHA-256 Digest: c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da
- Delta W: 0
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

CANONICAL_BASELINE_SHA256 = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
CANONICAL_PARAM_COUNT = 3443136


def compute_canonical_model_hash(model: nn.Module) -> Tuple[str, int, int]:
    """
    Computes canonical SHA-256 digest, parameter count, and total weight byte length.
    Returns: (sha256_hexdigest, total_params, total_weight_bytes)
    """
    hasher = hashlib.sha256()
    named_params = sorted(model.named_parameters(), key=lambda x: x[0])
    total_params = 0
    total_bytes = 0

    with torch.no_grad():
        for name, param in named_params:
            name_bytes = name.encode("utf-8")
            data_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(name_bytes)
            hasher.update(data_bytes)
            total_params += param.numel()
            total_bytes += len(data_bytes)

    return hasher.hexdigest(), total_params, total_bytes


def verify_canonical_baseline(model: nn.Module) -> Dict[str, Any]:
    """
    Verifies that the provided model matches the canonical baseline bit-for-bit.
    """
    digest, total_params, total_bytes = compute_canonical_model_hash(model)
    is_match = (
        digest == CANONICAL_BASELINE_SHA256 and total_params == CANONICAL_PARAM_COUNT
    )
    return {
        "digest": digest,
        "expected_digest": CANONICAL_BASELINE_SHA256,
        "total_params": total_params,
        "expected_params": CANONICAL_PARAM_COUNT,
        "total_weight_bytes": total_bytes,
        "delta_w": 0 if is_match else -1,
        "is_bit_exact": is_match,
    }
