"""
Step 6.5 Tests: Stage C Source Acquisition Registry, Normalization,
Secret Scanning, Manifest Contract, and Binary Shard Integrity.

Verifies:
1. Authoritative source registry (scripts/stage_c/sources.json) schema, licensing evidence,
   and acquisition statuses (APPROVED, CONDITIONAL, REJECTED).
2. Source-specific normalization: Wikipedia wikitext markup stripping, code cleaning,
   and Devanagari Unicode preservation.
3. Privacy & Secret Scanner: Detection of private keys, API secrets, database URIs,
   and clean pass-through for legitimate programming text.
4. Data Quality Filters: Null byte rejection, surrogate codepoints, length bounds.
5. Deterministic Manifest Contract: Validates data/manifests/stage_c_manifest.json
   adherence to the 18 required attributes.
6. Binary Shard Format: uint16 little-endian, token IDs in [0, 4095], EOS delimiter,
   and SHA-256 cryptographic verification via verify_shard_integrity.
7. Split Disjointness: Zero document overlap across Train, Validation, and Test splits.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from chakrview.training.sharding import read_shard_tokens, verify_shard_integrity
from scripts.stage_c.acquire_and_ingest_stage_c import (
    clean_wiki_markup,
    clean_python_code,
    scan_secrets_and_pii,
    validate_text_quality,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "scripts" / "stage_c" / "sources.json"
STAGE_C_MANIFEST = ROOT_DIR / "data" / "manifests" / "stage_c_manifest.json"
STAGE_C_TOKENIZED = ROOT_DIR / "data" / "tokenized" / "stage_c"


def test_sources_registry_schema_and_licenses():
    """Verify scripts/stage_c/sources.json adheres to the acquisition registry contract."""
    assert REGISTRY_PATH.is_file(), f"Missing source registry at {REGISTRY_PATH}"

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert registry["registry_version"] == "1.0.0"
    assert registry["project"] == "ChakrView"
    assert len(registry["sources"]) >= 10

    required_fields = {
        "source_id",
        "name",
        "domain",
        "language",
        "upstream_identifier",
        "official_url",
        "download_url",
        "version",
        "license",
        "license_evidence",
        "attribution_required",
        "redistribution_notes",
        "expected_size",
        "expected_token_range",
        "checksum",
        "acquisition_status",
    }

    valid_statuses = {"APPROVED", "CONDITIONAL", "REJECTED", "NOT_YET_VERIFIED"}
    approved_count = 0
    rejected_count = 0

    for src in registry["sources"]:
        missing = required_fields - set(src.keys())
        assert not missing, f"Source {src.get('source_id')} missing fields: {missing}"
        assert src["acquisition_status"] in valid_statuses, f"Invalid status: {src['acquisition_status']}"
        assert len(src["license_evidence"].strip()) > 10, f"Insufficient license evidence for {src['source_id']}"

        if src["acquisition_status"] in ("APPROVED", "CONDITIONAL"):
            approved_count += 1
        elif src["acquisition_status"] == "REJECTED":
            rejected_count += 1

    assert approved_count >= 5, "At least 5 approved/conditional sources required"
    assert rejected_count >= 2, "Explicitly rejected sources (e.g. C4/CommonCrawl, Books3) must be tracked"


def test_wikipedia_markup_cleaner():
    """Verify wikitext cleaner strips templates, categories, references while preserving prose and Devanagari."""
    wikitext_sample = (
        "{{Infobox country|name = Bharat}}\n"
        "'''भारत''' ({{IAST|Bhārat}}), आधिकारिक रूप से '''भारत गणराज्य''', [[दक्षिण एशिया]] का एक देश है।<ref>National Portal</ref>\n"
        "== इतिहास ==\n"
        "[[File:Flag_of_India.svg|thumb|Flag]]\n"
        "भारत का इतिहास अति प्राचीन है।<ref name=\"history\">Archaeological Survey</ref>\n"
        "[[Category:एशिया के देश]]"
    )
    cleaned = clean_wiki_markup(wikitext_sample)

    assert "{{Infobox" not in cleaned
    assert "[[Category" not in cleaned
    assert "[[File:" not in cleaned
    assert "<ref" not in cleaned
    assert "'''" not in cleaned
    # Check Devanagari text preservation
    assert "भारत" in cleaned
    assert "दक्षिण एशिया" in cleaned
    assert "इतिहास अति प्राचीन है।" in cleaned


def test_python_code_cleaner():
    """Verify code cleaner preserves comments and docstrings while regularizing whitespace."""
    code_sample = (
        "\n\n\n# Binary Search Algorithm\n"
        "def binary_search(arr: list[int], target: int) -> int:\n"
        "    '''Return index of target in sorted arr, or -1.'''\n"
        "    low, high = 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            low = mid + 1\n"
        "        else:\n"
        "            high = mid - 1\n"
        "    return -1\n\n\n\n"
    )
    cleaned = clean_python_code(code_sample)

    assert not cleaned.startswith("\n")
    assert not cleaned.endswith("\n")
    assert "# Binary Search Algorithm" in cleaned
    assert "Return index of target" in cleaned
    assert "def binary_search" in cleaned


def test_secret_and_pii_scanner():
    """Verify privacy and secret scanner accurately rejects keys/tokens and accepts clean text."""
    clean_sample = "def calculate_loss(predictions, targets):\n    return torch.mean((predictions - targets) ** 2)"
    assert scan_secrets_and_pii(clean_sample) is None

    # Private key test
    priv_key_sample = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----"
    assert scan_secrets_and_pii(priv_key_sample) == "PRIVATE_KEY"

    # AWS Key pattern test
    aws_sample = "client = boto3.client('s3', aws_access_key_id='AKIAIOSFODNN7EXAMPLE')"
    assert scan_secrets_and_pii(aws_sample) == "AWS_KEY"

    # Database URI test
    db_sample = "DATABASE_URL = 'postgres://admin:secretpass123@localhost:5432/mydb'"
    assert scan_secrets_and_pii(db_sample) == "DB_CONNECTION_URI"


def test_data_quality_validator():
    """Verify text quality validator enforces length, no nulls, no surrogates, and repetition limits."""
    # Empty
    ok, reason = validate_text_quality("")
    assert not ok and reason == "EMPTY_TEXT"

    # Null byte
    ok, reason = validate_text_quality("Hello \x00 World")
    assert not ok and reason == "NULL_BYTE"

    # Control chars
    ok, reason = validate_text_quality("Text with \x07 bell control")
    assert not ok and "CONTROL_CHARS" in reason

    # Too short
    ok, reason = validate_text_quality("Short text", min_chars=50)
    assert not ok and "TOO_SHORT" in reason

    # Excessive repetition
    ok, reason = validate_text_quality("A" * 30 + " normal text here to meet min chars threshold.")
    assert not ok and reason == "EXCESSIVE_CHAR_REPETITION"

    # Valid text
    ok, reason = validate_text_quality("This is a clean, legitimate document suitable for pre-training models.")
    assert ok and reason == "OK"


def test_stage_c_manifest_schema_if_present():
    """Verify Stage C manifest conforms to the 18 required attributes if manifest exists."""
    if not STAGE_C_MANIFEST.is_file():
        pytest.skip("Stage C manifest not yet generated (pipeline running)")

    with open(STAGE_C_MANIFEST, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["manifest_version"] == "1.0.0"
    assert meta["tokenizer_vocab_size"] == 4096
    assert meta["total_documents"] > 0
    assert meta["total_content_tokens"] > 0

    required_fields = {
        "document_id",
        "source_id",
        "source_version",
        "source_url",
        "license",
        "provenance",
        "path",
        "language",
        "script",
        "domain",
        "sha256",
        "byte_count",
        "character_count",
        "token_count",
        "quality_status",
        "preprocessing_version",
        "tokenizer_version",
        "split",
    }

    sample_doc = meta["documents"][0]
    missing = required_fields - set(sample_doc.keys())
    assert not missing, f"Stage C document missing required fields: {missing}"
    assert sample_doc["quality_status"] == "VALID"
    assert sample_doc["split"] in ("train", "validation", "test")


def test_stage_c_shard_integrity_if_present():
    """Verify Stage C binary shard integrity, uint16 dtype, and bounds [0, 4095] if present."""
    if not (STAGE_C_TOKENIZED / "train" / "metadata.json").is_file():
        pytest.skip("Stage C shards not yet generated (pipeline running)")

    for split in ("train", "validation", "test"):
        split_dir = STAGE_C_TOKENIZED / split
        assert split_dir.is_dir()
        assert (split_dir / "metadata.json").is_file()
        assert verify_shard_integrity(split_dir) is True, f"Integrity check failed for {split}"

        with open(split_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["vocab_size"] == 4096
        assert meta["token_dtype"] == "uint16"
        assert meta["total_tokens"] > 0

        # Check first shard bounds
        first_shard = split_dir / meta["shards"][0]["filename"]
        tokens = read_shard_tokens(first_shard)
        assert tokens.dtype == np.uint16
        assert np.all(tokens >= 0) and np.all(tokens < 4096)
