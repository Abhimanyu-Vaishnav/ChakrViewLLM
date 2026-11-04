"""
scripts/diagnose_cbr_sft_06_tokenizer.py

Forensic audit of ChakrView tokenizer on technical entity strings:
- IPv4 addresses
- Ports
- Hex hashes & commit SHAs
- UUIDs
- File paths & extensions
- Python identifiers & symbols
- Round-trip fidelity: decode(encode(text)) == text
"""

import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import load_tokenizer_artifacts

def run_forensics():
    vocab_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    assert vocab_dir.exists(), f"Vocab directory missing: {vocab_dir}"

    tok, _ = load_tokenizer_artifacts(vocab_dir)

    test_cases = [
        # IP Addresses
        ("ipv4_local", "192.168.1.105"),
        ("ipv4_public", "172.24.11.99"),
        ("ipv4_loopback", "127.0.0.1"),
        ("ipv4_dns", "10.128.4.55"),
        # Ports
        ("port_http", "8080"),
        ("port_https", "7443"),
        ("port_redis", "6379"),
        ("port_postgres", "5433"),
        ("port_high", "9999"),
        # Hashes & IDs
        ("git_short_sha", "a619297f"),
        ("git_long_sha", "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"),
        ("uuid_fragment", "73a24642-cbe6-4725"),
        # Paths
        ("path_unix", "/var/log/nginx/access.log"),
        ("path_win", "C:\\Users\\admin\\config.json"),
        ("path_env", "configs/staging/db.env"),
        # Identifiers & variables
        ("py_var_const", "DATABASE_PORT"),
        ("py_func", "def calculate_checksum(data: bytes) -> str:"),
        ("py_snake", "sensor_reading_raw"),
        ("py_camel", "TelemetryAccumulator"),
        # Symbols & Punctuation
        ("kv_pair", "MAX_CLIENTS=1024"),
        ("symbols", "foo_bar-baz.qux:80"),
        ("brackets", "{\"status\": \"healthy\", \"code\": 200}")
    ]

    results = []
    all_round_trips_pass = True

    for cat, text in test_cases:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        round_trip = (decoded == text)
        if not round_trip:
            all_round_trips_pass = False

        # Character boundaries per token
        token_pieces = []
        for tid in ids:
            try:
                token_pieces.append(tok.decode([tid]))
            except Exception:
                token_pieces.append(f"<byte_{tid}>")

        results.append({
            "category": cat,
            "original": text,
            "token_ids": ids,
            "token_count": len(ids),
            "token_pieces": token_pieces,
            "round_trip_success": round_trip,
            "decoded": decoded,
            "first_token_id": ids[0] if ids else None,
            "first_token_piece": token_pieces[0] if token_pieces else None
        })

    out_file = REPO_ROOT / "evaluation" / "cbr_sft_06_suite" / "tokenizer_forensics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_dir": str(vocab_dir),
            "all_round_trip_pass": all_round_trips_pass,
            "total_tested": len(test_cases),
            "cases": results
        }, f, indent=2)

    print(f"Tokenizer forensics complete. Total: {len(test_cases)}, All roundtrip pass: {all_round_trips_pass}")
    return results

if __name__ == "__main__":
    run_forensics()
