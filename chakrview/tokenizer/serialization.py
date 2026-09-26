"""
ChakrView Tokenizer: Transparent Serialization & Deserialization Engine.

Strictly fulfills the Section 20 Serialization Contract:
- vocab.json (token_id -> hex-encoded byte string for lossless binary representation)
- merges.json (pair_string "p0,p1" -> new_token_id)
- config.json (tokenizer_version, requested_vocab_size, actual_vocab_size,
               special_tokens, numeric_strategy, pretokenization_strategy,
               tie_breaking_rule, sha256 checksums, creation timestamp)

Zero opaque binary or pickling formats. All artifacts are directly inspectable.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from chakrview.tokenizer.bytes import (
    BYTE_OFFSET,
    NUM_BYTE_TOKENS,
)
from chakrview.tokenizer.special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    NUM_SPECIAL_TOKENS,
)
from chakrview.tokenizer.tokenizer import BPETokenizer

TOKENIZER_SERIALIZATION_VERSION = "0.1.0"


def save_tokenizer_artifacts(
    tokenizer: BPETokenizer,
    output_dir: Union[str, Path],
    numeric_strategy: str = "C",
    pretokenization_strategy: str = "byte",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save complete tokenizer artifacts into transparent, inspectable JSON files.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Save merges.json
    merges_dict = {f"{p[0]},{p[1]}": tid for p, tid in tokenizer.merges.items()}
    merges_content = json.dumps(merges_dict, indent=2)
    merges_path = out_p / "merges.json"
    merges_path.write_text(merges_content, encoding="utf-8")
    merges_sha = hashlib.sha256(merges_content.encode("utf-8")).hexdigest()

    # 2. Save vocab.json (Hex encoded bytes for non-printable octets)
    vocab_dict = {str(tid): b.hex() for tid, b in tokenizer.vocab.items()}
    vocab_content = json.dumps(vocab_dict, indent=2)
    vocab_path = out_p / "vocab.json"
    vocab_path.write_text(vocab_content, encoding="utf-8")
    vocab_sha = hashlib.sha256(vocab_content.encode("utf-8")).hexdigest()

    # 3. Save config.json
    cfg: Dict[str, Any] = {
        "format": "chakrview_bpe",
        "version": TOKENIZER_SERIALIZATION_VERSION,
        "vocab_size": tokenizer.vocab_size,
        "base_bytes_count": NUM_BYTE_TOKENS,
        "special_tokens_count": NUM_SPECIAL_TOKENS,
        "merges_count": len(tokenizer.merges),
        "special_tokens": {
            "<BOS>": BOS_ID,
            "<EOS>": EOS_ID,
            "<PAD>": PAD_ID,
        },
        "byte_offset": BYTE_OFFSET,
        "numeric_strategy": numeric_strategy,
        "pretokenization_strategy": pretokenization_strategy,
        "tie_breaking_rule": "(-frequency, pair[0], pair[1])",
        "checksums": {
            "merges_sha256": merges_sha,
            "vocab_sha256": vocab_sha,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if metadata:
        cfg["metadata"] = metadata

    config_content = json.dumps(cfg, indent=2)
    (out_p / "config.json").write_text(config_content, encoding="utf-8")

    return out_p


def load_tokenizer_artifacts(
    model_dir: Union[str, Path],
    verify_checksums: bool = True,
) -> Tuple[BPETokenizer, Dict[str, Any]]:
    """
    Load tokenizer from artifacts directory and verify integrity.

    Returns:
        (tokenizer_instance, config_dict)
    """
    p = Path(model_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {p}")

    cfg_path = p / "config.json"
    merges_path = p / "merges.json"
    vocab_path = p / "vocab.json"

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {p}")
    if not merges_path.is_file():
        raise FileNotFoundError(f"Missing merges.json in {p}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing vocab.json in {p}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    merges_raw_text = merges_path.read_text(encoding="utf-8")
    if verify_checksums and "checksums" in cfg:
        merges_sha = hashlib.sha256(merges_raw_text.encode("utf-8")).hexdigest()
        expected = cfg["checksums"].get("merges_sha256")
        if expected and merges_sha != expected:
            raise ValueError(f"merges.json checksum mismatch in {p}")

    raw_merges = json.loads(merges_raw_text)
    merges: Dict[Tuple[int, int], int] = {}
    for pair_str, tid in raw_merges.items():
        p0_str, p1_str = pair_str.split(",")
        merges[(int(p0_str), int(p1_str))] = int(tid)

    vocab_raw_text = vocab_path.read_text(encoding="utf-8")
    if verify_checksums and "checksums" in cfg:
        vocab_sha = hashlib.sha256(vocab_raw_text.encode("utf-8")).hexdigest()
        expected = cfg["checksums"].get("vocab_sha256")
        if expected and vocab_sha != expected:
            raise ValueError(f"vocab.json checksum mismatch in {p}")

    raw_vocab = json.loads(vocab_raw_text)
    vocab: Dict[int, bytes] = {
        int(tid): bytes.fromhex(hex_str) for tid, hex_str in raw_vocab.items()
    }

    tokenizer = BPETokenizer(merges=merges, vocab=vocab)
    return tokenizer, cfg
