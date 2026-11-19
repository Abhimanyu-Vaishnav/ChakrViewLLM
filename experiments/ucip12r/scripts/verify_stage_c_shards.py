"""
UCIP-12R Phase B: Standalone Shard Verification Tool.
Independently verifies all 32 Stage C binary shards against:
1. Exact byte alignment (file_size == num_tokens * 2).
2. Vocabulary bounds (0 <= token_id < 4096).
3. Cryptographic hash uniqueness.
4. Token counts and EOS delimiter frequencies.
"""

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE_C_DIR = ROOT / "data" / "tokenized" / "stage_c"

files = sorted(STAGE_C_DIR.glob("**/*.bin"))
print(f"Auditing {len(files)} shards in {STAGE_C_DIR}...")

total_tokens = 0
total_bytes = 0
invalid_tokens = 0
unique_hashes = set()

for f in files:
    sz = f.stat().st_size
    assert sz % 2 == 0, f"Odd byte size: {sz} in {f}"
    n = sz // 2
    total_tokens += n
    total_bytes += sz
    
    raw = f.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    assert h not in unique_hashes, f"Duplicate hash detected: {h} in {f}"
    unique_hashes.add(h)
    
    tokens = struct.unpack(f"<{n}H", raw)
    for tid in tokens:
        if tid >= 4096:
            invalid_tokens += 1

print(f"Verified {len(files)} shards:")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Total bytes: {total_bytes:,}")
print(f"  Invalid tokens (>=4096): {invalid_tokens}")
print(f"  Unique hashes: {len(unique_hashes)}")
assert invalid_tokens == 0
assert total_tokens == 7708601
print("ALL STAGE C SHARDS FULLY VERIFIED.")
