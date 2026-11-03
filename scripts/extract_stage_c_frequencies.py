"""
Standalone Stage C Frequency Extraction Runner for CAND_01.
Candidate: CAND_01_CONTROLLED_ENTITY_COPYING
Repository: ChakrView

GOVERNANCE RESTRICTIONS:
- Dedicated to deterministic token frequency counting across authorized Stage C training binary shards.
- Strictly operates on authorized Stage C training binary shards (shard_00000.bin .. shard_00026.bin).
- Prohibits access to Stage C validation/test splits, Stage B, or raw data.
- Enforces strict binary read-only I/O assertions (f.readable() and not f.writable()).
- Verifies SHA-256 integrity of input shards against metadata before processing.
- Implements v1.1 vocabulary decoding and filtering contract (hex decoding, replacement check, min length 3, alnum, no whitespace/delimiters, duplicate string mutual disqualification).
- Mandates pre-flight candidate pool margin assertion (len(eligible_tokens) >= 600).
- Applies deterministic integer frequency-tier allocation (N1 = N//3, N2 = N//3, N3 = N - N1 - N2).
- Enforces tie-breaking on equal counts via ascending Token ID.
- Writes atomic outputs and complete cryptographic manifests to isolated non-canonical directories.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import psutil

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Authoritative Governance Constants
REGISTERED_STAGE_C_TRAIN_SHARD_COUNT = 27
REGISTERED_STAGE_C_TRAIN_TOTAL_TOKENS = 6651093
REGISTERED_VOCAB_SIZE = 4096
MIN_ELIGIBLE_TOKENS = 600
PROPOSED_RAM_CEILING_MB = 512.0
PROPOSED_TIMEOUT_SECONDS = 60.0

EXPECTED_AUTHORIZATION_DECISION_DOC = "docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md"
EXPECTED_AUTHORIZATION_DECISION_HASH = "e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00"

FORBIDDEN_OUTPUT_SUBSTRINGS = [
    "data/tokenized",
    "data/preprocessed",
    "data/tokenizer",
    "data/manifests",
    "chakrview",
    "models",
    "checkpoints",
    "experiments/overnight_runs",
]


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in binary mode."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_output_directory(output_dir: Path) -> None:
    """Verifies output path does not collide with canonical or protected directories."""
    resolved_out = output_dir.resolve()
    norm_resolved = str(resolved_out).replace("\\", "/").lower()

    for forbidden in FORBIDDEN_OUTPUT_SUBSTRINGS:
        forbidden_target = (REPO_ROOT / forbidden).resolve()
        norm_target = str(forbidden_target).replace("\\", "/").lower()
        if norm_resolved == norm_target or norm_resolved.startswith(norm_target + "/"):
            raise PermissionError(
                f"FATAL GOVERNANCE BREACH: Output directory '{output_dir}' intersects protected path '{forbidden}'."
            )


def normalize_and_validate_path(path_str: str, param_name: str, allow_existing: bool = True) -> Path:
    """
    Normalizes path, rejects traversal, symlink escapes, and checks forbidden directory targets.
    """
    if not path_str:
        raise ValueError(f"Parameter '{param_name}' must not be empty.")
    
    raw_p = Path(path_str)
    if ".." in raw_p.parts:
        raise ValueError(f"FATAL: Path traversal forbidden in '{param_name}': {path_str}")

    p = raw_p.resolve()
    norm_str = str(p).replace("\\", "/").lower()

    # Disallow overnight runs under all circumstances
    if "experiments/overnight_runs" in norm_str:
        raise PermissionError(f"FATAL GOVERNANCE BREACH: '{param_name}' attempts access to excluded overnight_runs.")

    if "output" in param_name.lower():
        validate_output_directory(p)

    return p


def decode_vocab_entry(hex_val: str) -> str:
    """
    Decodes hexadecimal vocabulary string per the v1.1 decoding contract:
    bytes.fromhex(val).decode('utf-8', errors='replace')
    """
    try:
        raw_b = bytes.fromhex(hex_val)
    except Exception as e:
        raise ValueError(f"Malformed hexadecimal token string '{hex_val}': {e}")
    return raw_b.decode("utf-8", errors="replace")


def is_candidate_entity_token(token_id: int, decoded_token: str) -> bool:
    """
    Evaluates lexical criteria per CAND_01 v1.1 specification:
    - ID >= 259 (IDs 0..2 reserved, IDs 3..258 single-byte tokens excluded)
    - Replacement character (\ufffd) excluded
    - Length >= 3 characters
    - Purely alphanumeric (isalnum())
    - No whitespace or control characters
    """
    if token_id < 259:
        return False
    if "\ufffd" in decoded_token:
        return False
    if len(decoded_token) < 3:
        return False
    if not decoded_token.isalnum():
        return False
    if any(c.isspace() or ord(c) < 32 for c in decoded_token):
        return False
    return True


def filter_and_validate_vocabulary(
    vocab_dict: Dict[str, str],
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Filters vocabulary dictionary according to the v1.1 contract:
    - Decodes hex strings
    - Applies lexical filters
    - Detects duplicate decoded strings and mutually disqualifies them
    - Returns eligible token mapping {token_id: decoded_string} and audit metadata.
    """
    decoded_entries: Dict[int, str] = {}
    lexical_candidates: Dict[int, str] = {}
    rejection_reasons: Dict[str, int] = {
        "reserved_id": 0,
        "hex_decode_error": 0,
        "unicode_replacement": 0,
        "length_under_3": 0,
        "not_alphanumeric": 0,
        "whitespace_or_control": 0,
        "duplicate_string": 0,
    }

    for tok_id_str, hex_str in vocab_dict.items():
        tok_id = int(tok_id_str)
        if tok_id < 259:
            rejection_reasons["reserved_id"] += 1
            continue
        try:
            dec = decode_vocab_entry(hex_str)
        except ValueError:
            rejection_reasons["hex_decode_error"] += 1
            continue

        decoded_entries[tok_id] = dec

        if "\ufffd" in dec:
            rejection_reasons["unicode_replacement"] += 1
            continue
        if len(dec) < 3:
            rejection_reasons["length_under_3"] += 1
            continue
        if not dec.isalnum():
            rejection_reasons["not_alphanumeric"] += 1
            continue
        if any(c.isspace() or ord(c) < 32 for c in dec):
            rejection_reasons["whitespace_or_control"] += 1
            continue

        lexical_candidates[tok_id] = dec

    # Duplicate decoded string mutual disqualification
    from collections import Counter
    string_counts = Counter(lexical_candidates.values())
    duplicate_strings = {s for s, count in string_counts.items() if count > 1}

    eligible_tokens: Dict[int, str] = {}
    for tid, s in lexical_candidates.items():
        if s in duplicate_strings:
            rejection_reasons["duplicate_string"] += 1
        else:
            eligible_tokens[tid] = s

    audit_info = {
        "total_vocab_entries": len(vocab_dict),
        "surviving_eligible_count": len(eligible_tokens),
        "rejection_breakdown": rejection_reasons,
        "duplicate_strings_count": len(duplicate_strings),
    }

    return eligible_tokens, audit_info


def allocate_frequency_tiers(
    candidate_counts: List[Tuple[int, int, str]],
) -> Dict[str, Any]:
    """
    Applies frozen integer frequency-tier allocation:
    N1 = floor(N / 3) (Tier 1: High Frequency)
    N2 = floor(N / 3) (Tier 2: Medium Frequency)
    N3 = N - N1 - N2  (Tier 3: Low Frequency)
    Sorted descending by count (-count), with ascending Token ID (token_id) tie-breaking.
    candidate_counts: List of (token_id, count, token_str)
    """
    # Exclude zero-frequency entries
    non_zero_candidates = [c for c in candidate_counts if c[1] > 0]
    n = len(non_zero_candidates)
    zero_freq_count = len(candidate_counts) - n

    assert n >= MIN_ELIGIBLE_TOKENS, (
        f"FATAL: Candidate pool margin failure! Required >= {MIN_ELIGIBLE_TOKENS}, found {n}."
    )

    # Sort descending by count, ascending by token_id
    sorted_candidates = sorted(non_zero_candidates, key=lambda x: (-x[1], x[0]))

    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2

    tier1 = sorted_candidates[:n1]
    tier2 = sorted_candidates[n1:n1 + n2]
    tier3 = sorted_candidates[n1 + n2:]

    assert len(tier1) == n1
    assert len(tier2) == n2
    assert len(tier3) == n3
    assert len(tier1) + len(tier2) + len(tier3) == n

    # Sub-select experimental pools: 450 train (150/tier), 150 test (50/tier)
    e_train = tier1[:150] + tier2[:150] + tier3[:150]
    e_test = tier1[150:200] + tier2[150:200] + tier3[150:200]

    train_ids = {x[0] for x in e_train}
    test_ids = {x[0] for x in e_test}

    assert len(train_ids) == 450, f"Expected 450 train tokens, got {len(train_ids)}"
    assert len(test_ids) == 150, f"Expected 150 test tokens, got {len(test_ids)}"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test pools must be strictly disjoint!"

    return {
        "total_eligible_non_zero": n,
        "zero_frequency_count": zero_freq_count,
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "tier1_tokens": [{"token_id": t[0], "count": t[1], "token": t[2]} for t in tier1],
        "tier2_tokens": [{"token_id": t[0], "count": t[1], "token": t[2]} for t in tier2],
        "tier3_tokens": [{"token_id": t[0], "count": t[1], "token": t[2]} for t in tier3],
        "e_train_pool": [{"token_id": t[0], "count": t[1], "token": t[2]} for t in e_train],
        "e_test_pool": [{"token_id": t[0], "count": t[1], "token": t[2]} for t in e_test],
    }


def execute_frequency_extraction(
    input_dir: Path,
    vocab_path: Path,
    output_dir: Path,
    expected_shards: Optional[int] = None,
    expected_tokens: Optional[int] = None,
    vocab_size: int = REGISTERED_VOCAB_SIZE,
    max_timeout_seconds: float = PROPOSED_TIMEOUT_SECONDS,
    max_ram_mb: float = PROPOSED_RAM_CEILING_MB,
    synthetic_mode: bool = False,
    authorization_decision_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes frequency extraction with strict read-only guarantees, SHA-256 pre-verification,
    real timeout and memory enforcement, telemetry recording, and atomic output serialization.

    Authorization Gate:
    - If synthetic_mode is True, execution proceeds safely under mock/synthetic test fixtures.
    - If synthetic_mode is False (production mode), authorization_decision_ref must be supplied
      and point to a valid, existing authorization decision document matching the expected
      governance criteria.
    """
    verified_auth_doc: Optional[Path] = None
    if not synthetic_mode:
        if not authorization_decision_ref:
            raise PermissionError(
                "FATAL GOVERNANCE BREACH: Physical Stage C Frequency Extraction requires explicit authorization!\n"
                "execute_frequency_extraction() with synthetic_mode=False requires authorization_decision_ref "
                "referencing 'CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md'."
            )
        # Normalize and verify authorization decision document reference
        raw_ref = Path(authorization_decision_ref)
        if ".." in raw_ref.parts:
            raise ValueError(f"FATAL: Path traversal forbidden in authorization_decision_ref: {authorization_decision_ref}")
        
        # Deterministic Canonical Document Target
        canonical_doc = (REPO_ROOT / EXPECTED_AUTHORIZATION_DECISION_DOC).resolve()
        if not canonical_doc.exists() or not canonical_doc.is_file():
            raise FileNotFoundError(f"FATAL: Canonical authorization document missing at {canonical_doc}")

        # Resolve candidate reference
        if raw_ref.is_absolute():
            resolved_ref = raw_ref.resolve()
        else:
            resolved_ref = (REPO_ROOT / raw_ref).resolve()

        # Strict identity check: Must resolve to the exact canonical repository file
        if resolved_ref != canonical_doc:
            raise PermissionError(
                f"FATAL GOVERNANCE BREACH: Authorization decision reference '{authorization_decision_ref}'\n"
                f"does not resolve to the canonical repository path:\n"
                f"Expected: {canonical_doc}\n"
                f"Resolved: {resolved_ref}\n"
                f"External copies or alternate directory locations are strictly rejected."
            )
        
        # Verify content integrity of the canonical authorization decision document
        doc_hash = compute_file_sha256(canonical_doc)
        if doc_hash != EXPECTED_AUTHORIZATION_DECISION_HASH:
            raise PermissionError(
                f"FATAL GOVERNANCE BREACH: Authorization decision document checksum mismatch!\n"
                f"Expected SHA-256: {EXPECTED_AUTHORIZATION_DECISION_HASH}\n"
                f"Actual SHA-256:   {doc_hash}"
            )
        verified_auth_doc = canonical_doc

    start_time = time.perf_counter()
    start_cpu = time.process_time()
    proc = psutil.Process()

    # 1. Output Path Safety Verification
    validate_output_directory(output_dir)
    if output_dir.exists():
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            raise FileExistsError(
                f"FATAL: Output directory '{output_dir}' already contains JSON artifacts! Overwriting prohibited."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Vocabulary Parsing & Filtering
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    eligible_tokens, vocab_audit = filter_and_validate_vocabulary(vocab_dict)

    if len(eligible_tokens) < MIN_ELIGIBLE_TOKENS:
        raise ValueError(
            f"FATAL: Insufficient eligible tokens: {len(eligible_tokens)} < {MIN_ELIGIBLE_TOKENS} required."
        )

    # 3. Read & Verify Metadata JSON (accepts stage_c_metadata.json or metadata.json)
    meta_path = input_dir / "stage_c_metadata.json"
    if not meta_path.exists():
        meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {input_dir / 'metadata.json'}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Metadata schema assertions
    assert metadata.get("vocab_size") == vocab_size, f"Vocab size mismatch in metadata: {metadata.get('vocab_size')}"
    assert metadata.get("token_dtype") == "uint16", f"Unsupported token dtype: {metadata.get('token_dtype')}"
    shards_list = metadata.get("shards", [])
    if expected_shards is not None:
        assert len(shards_list) == expected_shards, (
            f"Shard count mismatch: metadata has {len(shards_list)}, expected {expected_shards}"
        )

    # Reject unexpected or extraneous .bin files not recorded in metadata
    disk_bins = {p.name for p in input_dir.glob("*.bin")}
    meta_bins = {s["filename"] for s in shards_list}
    extraneous = disk_bins - meta_bins
    if extraneous:
        raise ValueError(f"Extraneous .bin files discovered in input directory: {extraneous}")

    # 4. Pre-Verification of Shard Checksums & Token Counting
    counts = np.zeros(vocab_size, dtype=np.int64)
    total_tokens_read = 0
    shard_telemetry = []

    for s_info in shards_list:
        # Check timeout ceiling enforcement
        elapsed = time.perf_counter() - start_time
        if elapsed > max_timeout_seconds:
            raise TimeoutError(f"Extraction exceeded maximum timeout ceiling of {max_timeout_seconds}s (elapsed: {elapsed:.2f}s)!")

        # Check memory ceiling enforcement
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        if rss_mb > max_ram_mb:
            raise MemoryError(f"Extraction exceeded maximum RAM ceiling of {max_ram_mb} MB (current RSS: {rss_mb:.2f} MB)!")

        shard_name = s_info["filename"]
        expected_sha = s_info["sha256"]
        expected_bytes = s_info["byte_size"]
        expected_shard_tokens = s_info["token_count"]

        shard_path = input_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected shard missing on disk: {shard_path}")

        # Check byte size without reading token array
        actual_bytes = shard_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"Byte size mismatch for {shard_name}: expected {expected_bytes}, got {actual_bytes}")
        if actual_bytes % 2 != 0:
            raise ValueError(f"Truncated uint16 binary file {shard_name}: byte size {actual_bytes} is not even!")

        # Pre-execution SHA-256 Checksum Verification
        actual_sha = compute_file_sha256(shard_path)
        if actual_sha != expected_sha:
            raise ValueError(f"Cryptographic hash mismatch on {shard_name}: expected {expected_sha}, got {actual_sha}")

        # Read Shard under Strict Read-Only Contract
        with open(shard_path, "rb") as f:
            assert f.readable(), f"File handle not readable: {shard_path}"
            assert not f.writable(), f"File handle acquired write permissions: {shard_path}"
            shard_bytes = f.read()

        shard_tokens = np.frombuffer(shard_bytes, dtype="<u2")
        if len(shard_tokens) != expected_shard_tokens:
            raise ValueError(f"Token count mismatch for {shard_name}: expected {expected_shard_tokens}, got {len(shard_tokens)}")

        # Range check on token IDs
        if np.any(shard_tokens >= vocab_size):
            max_found = int(np.max(shard_tokens))
            raise ValueError(f"Out-of-range token ID {max_found} encountered in {shard_name} (vocab_size={vocab_size})")

        # Accumulate frequency
        shard_counts = np.bincount(shard_tokens, minlength=vocab_size)
        counts += shard_counts
        total_tokens_read += len(shard_tokens)

        shard_telemetry.append({
            "shard": shard_name,
            "tokens": len(shard_tokens),
            "sha256": actual_sha,
        })

    # Validate overall count reconciliation
    if expected_tokens is not None and total_tokens_read != expected_tokens:
        raise ValueError(
            f"FATAL: Total tokens counted ({total_tokens_read}) does not match expected total ({expected_tokens})!"
        )

    # 5. Extract Candidate Counts & Stratify Tiers
    candidate_counts = [
        (tid, int(counts[tid]), eligible_tokens[tid]) for tid in eligible_tokens
    ]
    tier_results = allocate_frequency_tiers(candidate_counts)

    elapsed_wall = time.perf_counter() - start_time
    elapsed_cpu = time.process_time() - start_cpu
    final_rss_mb = proc.memory_info().rss / (1024 * 1024)

    # 6. Atomic Serialization to Output Directory
    tmp_out = output_dir / ".extraction_tmp"
    tmp_out.mkdir(parents=True, exist_ok=True)

    freq_path = tmp_out / "token_frequencies_stage_c_train.json"
    pools_path = tmp_out / "candidate_entity_pools.json"
    manifest_path = tmp_out / "extraction_manifest.json"

    # Serialize Frequency Mapping
    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump({str(i): int(counts[i]) for i in range(vocab_size)}, f, indent=2)

    # Serialize Candidate Entity Pools
    with open(pools_path, "w", encoding="utf-8") as f:
        json.dump(tier_results, f, indent=2)

    # Serialize Manifest & Telemetry
    manifest_payload = {
        "manifest_version": "1.0.0",
        "synthetic_mode": synthetic_mode,
        "authorization_decision_ref": str(verified_auth_doc) if verified_auth_doc else None,
        "input_directory": str(input_dir),
        "vocab_path": str(vocab_path),
        "source_metadata_hash": compute_file_sha256(meta_path),
        "shards_processed": len(shard_telemetry),
        "total_tokens_counted": total_tokens_read,
        "vocabulary_audit": vocab_audit,
        "tier_partition_summary": {
            "eligible_candidates_non_zero": tier_results["total_eligible_non_zero"],
            "tier1_count": tier_results["n1"],
            "tier2_count": tier_results["n2"],
            "tier3_count": tier_results["n3"],
            "e_train_count": len(tier_results["e_train_pool"]),
            "e_test_count": len(tier_results["e_test_pool"]),
        },
        "telemetry": {
            "wall_clock_seconds": elapsed_wall,
            "cpu_time_seconds": elapsed_cpu,
            "rss_memory_mb": final_rss_mb,
            "python_version": sys.version,
        },
        "output_hashes": {
            "token_frequencies_sha256": compute_file_sha256(freq_path),
            "candidate_entity_pools_sha256": compute_file_sha256(pools_path),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # Atomic move
    for fname in ["token_frequencies_stage_c_train.json", "candidate_entity_pools.json", "extraction_manifest.json"]:
        os.replace(tmp_out / fname, output_dir / fname)
    if tmp_out.exists():
        tmp_out.rmdir()

    return manifest_payload


def parse_args():
    parser = argparse.ArgumentParser(description="CAND_01 Stage C Frequency Extraction Runner")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to Stage C training shards directory")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocab_4096/vocab.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write frequency artifacts")
    parser.add_argument("--synthetic", action="store_true", help="Execute against synthetic mock shards (safe testing mode)")
    parser.add_argument("--execute-production", action="store_true", help="Explicit intent to run authorized production extraction")
    parser.add_argument("--authorization-decision-ref", type=str, default=None, help="Reference to signed CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md")
    return parser.parse_args()


def main():
    args = parse_args()

    # Mutual exclusivity and authorization verification
    if args.synthetic and args.execute_production:
        sys.stderr.write(
            "FATAL CONFIGURATION ERROR: --synthetic and --execute-production are mutually exclusive.\n"
        )
        sys.exit(1)

    if not args.synthetic and not args.execute_production:
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: Explicit execution mode required.\n"
            "Specify --synthetic for safe testing, or --execute-production with --authorization-decision-ref for production.\n"
        )
        sys.exit(1)

    if args.execute_production:
        if not args.authorization_decision_ref:
            sys.stderr.write(
                "FATAL GOVERNANCE BREACH: --execute-production requires --authorization-decision-ref referencing\n"
                "'CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md'.\n"
            )
            sys.exit(1)
        mode_label = "PRODUCTION MODE (AUTHORIZED)"
        is_synthetic = False
    else:
        mode_label = "SYNTHETIC MODE"
        is_synthetic = True

    print("=================================================================")
    print(f"CAND_01 STAGE C FREQUENCY EXTRACTION RUNNER ({mode_label})")
    print(f"Input: {args.input_dir} | Output: {args.output_dir}")
    print("=================================================================")

    in_dir = normalize_and_validate_path(args.input_dir, "--input-dir")
    v_path = normalize_and_validate_path(args.vocab_path, "--vocab-path")
    out_dir = normalize_and_validate_path(args.output_dir, "--output-dir")

    manifest = execute_frequency_extraction(
        input_dir=in_dir,
        vocab_path=v_path,
        output_dir=out_dir,
        synthetic_mode=is_synthetic,
        authorization_decision_ref=args.authorization_decision_ref if not is_synthetic else None,
    )

    print("\nExtraction Summary:")
    print(f"  Execution Mode:   {mode_label}")
    print(f"  Shards Processed: {manifest['shards_processed']}")
    print(f"  Tokens Counted:   {manifest['total_tokens_counted']}")
    print(f"  Eligible Tokens:  {manifest['tier_partition_summary']['eligible_candidates_non_zero']}")
    print(f"  Wall-Clock Time:  {manifest['telemetry']['wall_clock_seconds']:.4f}s")
    print(f"  CPU Time:         {manifest['telemetry']['cpu_time_seconds']:.4f}s")
    print(f"\nSUCCESS: Extraction runner ({mode_label}) executed cleanly.")


if __name__ == "__main__":
    main()
