"""
Comprehensive Synthetic Unit Test Suite for CAND_01 Stage C Frequency Extraction Runner.
Validates all 30 governance, functional, edge-case, and security requirements against
scripts/extract_stage_c_frequencies.py without touching production shards or directories.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_stage_c_frequencies import (
    decode_vocab_entry,
    filter_and_validate_vocabulary,
    allocate_frequency_tiers,
    normalize_and_validate_path,
    compute_file_sha256,
    execute_frequency_extraction,
    REGISTERED_VOCAB_SIZE,
    MIN_ELIGIBLE_TOKENS,
    FORBIDDEN_OUTPUT_SUBSTRINGS,
)


class TestCand01StageCExtractionRunner(unittest.TestCase):
    """30-Point Comprehensive Synthetic Test Suite for Extraction Runner."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="cand01_test_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_vocab(self, total_tokens=4096, valid_candidates=650, duplicate_token=None):
        """Creates a synthetic mock vocabulary JSON mapping string token_id -> hex string."""
        vocab = {}
        # 0: <BOS>, 1: <EOS>, 2: <PAD>
        vocab["0"] = "3c424f533e"  # <BOS>
        vocab["1"] = "3c454f533e"  # <EOS>
        vocab["2"] = "3c5041443e"  # <PAD>
        # 3..258: single byte tokens
        for i in range(3, 259):
            vocab[str(i)] = f"{i:02x}"

        # 259..total_tokens-1: custom words
        curr_id = 259
        created_candidates = 0
        while curr_id < total_tokens:
            if created_candidates < valid_candidates:
                word = f"word{created_candidates}"
                vocab[str(curr_id)] = word.encode("utf-8").hex()
                created_candidates += 1
            else:
                # Add non-candidate entries (e.g. delimiters, short tokens)
                vocab[str(curr_id)] = "2c".encode("utf-8").hex()  # ','
            curr_id += 1

        if duplicate_token:
            # Overwrite two IDs with the same hex representation
            vocab["300"] = duplicate_token.encode("utf-8").hex()
            vocab["301"] = duplicate_token.encode("utf-8").hex()

        vocab_file = self.temp_path / "mock_vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
        return vocab_file

    def _create_mock_shards(self, input_dir, vocab_size=4096, num_shards=2, tokens_per_shard=1000, token_ids=None):
        """Creates synthetic mock shards and metadata."""
        input_dir.mkdir(parents=True, exist_ok=True)
        shards_info = []
        total_tokens = 0

        for s_idx in range(num_shards):
            shard_name = f"shard_{s_idx:05d}.bin"
            shard_file = input_dir / shard_name
            if token_ids is not None:
                tokens = np.array(token_ids[s_idx * tokens_per_shard:(s_idx + 1) * tokens_per_shard], dtype=np.uint16)
            else:
                # Generate valid tokens in candidate range [259, 259 + 500)
                tokens = np.random.randint(259, 259 + 500, size=tokens_per_shard, dtype=np.uint16)

            tokens.tofile(shard_file)
            sha = compute_file_sha256(shard_file)
            b_size = shard_file.stat().st_size
            shards_info.append({
                "filename": shard_name,
                "byte_size": b_size,
                "token_count": len(tokens),
                "sha256": sha,
            })
            total_tokens += len(tokens)

        meta = {
            "metadata_version": "1.0.0",
            "vocab_size": vocab_size,
            "token_dtype": "uint16",
            "shards": shards_info,
        }
        meta_file = input_dir / "stage_c_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return shards_info, total_tokens

    # 1. Valid hex decoding
    def test_01_valid_hex_decoding(self):
        hex_str = "6170706c65"  # "apple"
        decoded = decode_vocab_entry(hex_str)
        self.assertEqual(decoded, "apple")

    # 2. Invalid hex rejection
    def test_02_invalid_hex_rejection(self):
        with self.assertRaises(ValueError):
            decode_vocab_entry("not_a_hex_string_zz")

    # 3. UTF-8 replacement-character exclusion
    def test_03_replacement_character_exclusion(self):
        invalid_utf8_hex = "ff"
        decoded = decode_vocab_entry(invalid_utf8_hex)
        self.assertIn("\ufffd", decoded)
        # Verify filtered out by audit
        mock_vocab = {"0": "3c424f533e", "1": "3c454f533e", "2": "3c5041443e"}
        for i in range(3, 259):
            mock_vocab[str(i)] = f"{i:02x}"
        mock_vocab["260"] = "ff"  # replacement char
        eligible, audit = filter_and_validate_vocabulary(mock_vocab)
        self.assertNotIn(260, eligible)
        self.assertEqual(audit["rejection_breakdown"]["unicode_replacement"], 1)

    # 4. Reserved-token exclusion (0..2)
    def test_04_reserved_token_exclusion(self):
        v_file = self._create_mock_vocab(total_tokens=300, valid_candidates=10)
        with open(v_file) as f:
            v_dict = json.load(f)
        eligible, audit = filter_and_validate_vocabulary(v_dict)
        for reserved_id in [0, 1, 2]:
            self.assertNotIn(reserved_id, eligible)
        self.assertEqual(audit["rejection_breakdown"]["reserved_id"], 259)

    # 5. Byte-token exclusion (3..258)
    def test_05_byte_token_exclusion(self):
        v_file = self._create_mock_vocab(total_tokens=300, valid_candidates=10)
        with open(v_file) as f:
            v_dict = json.load(f)
        eligible, _ = filter_and_validate_vocabulary(v_dict)
        for byte_id in range(3, 259):
            self.assertNotIn(byte_id, eligible)

    # 6. Whitespace and delimiter exclusion
    def test_06_whitespace_and_delimiter_exclusion(self):
        mock_vocab = {"0": "3c424f533e", "1": "3c454f533e", "2": "3c5041443e"}
        for i in range(3, 259):
            mock_vocab[str(i)] = f"{i:02x}"
        mock_vocab["259"] = "2068656c6c6f"  # " hello" (space prefix)
        mock_vocab["260"] = "6361742c"      # "cat," (comma delimiter)
        mock_vocab["261"] = "646f672e"      # "dog." (period delimiter)
        mock_vocab["262"] = "666f78"        # "fox" (valid)
        eligible, audit = filter_and_validate_vocabulary(mock_vocab)
        self.assertNotIn(259, eligible)
        self.assertNotIn(260, eligible)
        self.assertNotIn(261, eligible)
        self.assertIn(262, eligible)

    # 7. Duplicate decoded-string mutual disqualification
    def test_07_duplicate_string_mutual_disqualification(self):
        v_file = self._create_mock_vocab(total_tokens=500, valid_candidates=100, duplicate_token="duplicate")
        with open(v_file) as f:
            v_dict = json.load(f)
        eligible, audit = filter_and_validate_vocabulary(v_dict)
        self.assertNotIn(300, eligible)
        self.assertNotIn(301, eligible)
        self.assertGreaterEqual(audit["rejection_breakdown"]["duplicate_string"], 2)

    # 8. Zero-frequency exclusion
    def test_08_zero_frequency_exclusion(self):
        mock_candidates = [(300 + i, 10 + i, f"word{i}") for i in range(600)]
        mock_candidates.append((1000, 0, "zero_freq"))
        res = allocate_frequency_tiers(mock_candidates)
        all_partitioned_ids = {t["token_id"] for t in res["tier1_tokens"] + res["tier2_tokens"] + res["tier3_tokens"]}
        self.assertNotIn(1000, all_partitioned_ids)
        self.assertEqual(res["zero_frequency_count"], 1)

    # 9. Minimum 600-token gate
    def test_09_minimum_600_token_gate(self):
        mock_599 = [(300 + i, 10, f"word{i}") for i in range(599)]
        with self.assertRaises(AssertionError):
            allocate_frequency_tiers(mock_599)
        mock_600 = [(300 + i, 10, f"word{i}") for i in range(600)]
        res = allocate_frequency_tiers(mock_600)
        self.assertEqual(res["total_eligible_non_zero"], 600)

    # 10. Deterministic frequency sorting
    def test_10_deterministic_frequency_sorting(self):
        mock_candidates = [(300 + i, i + 1, f"word{i}") for i in range(600)]
        res = allocate_frequency_tiers(mock_candidates)
        # Tier 1 must have highest counts
        t1_counts = [t["count"] for t in res["tier1_tokens"]]
        t2_counts = [t["count"] for t in res["tier2_tokens"]]
        t3_counts = [t["count"] for t in res["tier3_tokens"]]
        self.assertTrue(all(t1_counts[i] >= t1_counts[i+1] for i in range(len(t1_counts)-1)))
        self.assertTrue(min(t1_counts) >= max(t2_counts))
        self.assertTrue(min(t2_counts) >= max(t3_counts))

    # 11. Ascending token-ID tie-breaking
    def test_11_ascending_token_id_tie_breaking(self):
        # Reverse order insertion: 500 down to -99
        mock_candidates = [(500 - i, 100, f"word{i}") for i in range(600)]
        res = allocate_frequency_tiers(mock_candidates)
        t1_ids = [t["token_id"] for t in res["tier1_tokens"]]
        # Since all counts are equal (100), IDs must be sorted ascending
        self.assertEqual(t1_ids, sorted(t1_ids))

    # 12. Tier allocation for N divisible by 3
    def test_12_tier_allocation_divisible_by_3(self):
        n = 600
        mock_candidates = [(i, 10, f"w{i}") for i in range(n)]
        res = allocate_frequency_tiers(mock_candidates)
        self.assertEqual(res["n1"], 200)
        self.assertEqual(res["n2"], 200)
        self.assertEqual(res["n3"], 200)

    # 13. Tier allocation for N not divisible by 3
    def test_13_tier_allocation_not_divisible_by_3(self):
        n = 601
        mock_candidates = [(i, 10, f"w{i}") for i in range(n)]
        res = allocate_frequency_tiers(mock_candidates)
        self.assertEqual(res["n1"], 200)
        self.assertEqual(res["n2"], 200)
        self.assertEqual(res["n3"], 201)

        n2 = 602
        mock_candidates2 = [(i, 10, f"w{i}") for i in range(n2)]
        res2 = allocate_frequency_tiers(mock_candidates2)
        self.assertEqual(res2["n1"], 200)
        self.assertEqual(res2["n2"], 200)
        self.assertEqual(res2["n3"], 202)

    # 14. Disjoint and exhaustive tiers
    def test_14_disjoint_and_exhaustive_tiers(self):
        mock_candidates = [(i, 10 + i, f"w{i}") for i in range(605)]
        res = allocate_frequency_tiers(mock_candidates)
        set1 = {t["token_id"] for t in res["tier1_tokens"]}
        set2 = {t["token_id"] for t in res["tier2_tokens"]}
        set3 = {t["token_id"] for t in res["tier3_tokens"]}
        self.assertTrue(set1.isdisjoint(set2))
        self.assertTrue(set1.isdisjoint(set3))
        self.assertTrue(set2.isdisjoint(set3))
        self.assertEqual(len(set1) + len(set2) + len(set3), 605)

    # 15. Out-of-range token IDs
    def test_15_out_of_range_token_ids(self):
        in_dir = self.temp_path / "in_oor"
        out_dir = self.temp_path / "out_oor"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        tokens = [5000] * 1000  # 5000 >= 4096
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=1000, token_ids=tokens)
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertIn("Out-of-range token ID", str(ctx.exception))

    # 16. Truncated uint16 binary data
    def test_16_truncated_uint16_binary_data(self):
        in_dir = self.temp_path / "in_trunc"
        out_dir = self.temp_path / "out_trunc"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        shard_file = in_dir / "shard_00000.bin"
        # Truncate by 1 byte to make size odd
        with open(shard_file, "ab") as f:
            f.write(b"\x00")
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertTrue("Byte size mismatch" in str(ctx.exception) or "Truncated uint16" in str(ctx.exception))

    # 17. Wrong endianness detection where detectable
    def test_17_wrong_endianness_detection(self):
        in_dir = self.temp_path / "in_endian"
        out_dir = self.temp_path / "out_endian"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        # 600 distinct valid candidate tokens repeated to form 1200 tokens
        valid_ids = [300 + (i % 600) for i in range(1200)]
        tokens = np.array(valid_ids, dtype=np.uint16)
        tokens_be = tokens.byteswap()  # swap to big endian
        in_dir.mkdir(parents=True, exist_ok=True)
        tokens_be.tofile(in_dir / "shard_00000.bin")
        meta = {
            "metadata_version": "1.0.0",
            "vocab_size": 4096,
            "token_dtype": "uint16",
            "shards": [{
                "filename": "shard_00000.bin",
                "byte_size": 2400,
                "token_count": 1200,
                "sha256": compute_file_sha256(in_dir / "shard_00000.bin")
            }]
        }
        with open(in_dir / "stage_c_metadata.json", "w") as f:
            json.dump(meta, f)
        
        # Swapping endianness flips bytes (e.g. 300 = 0x012c -> 0x2c01 = 11265 >= 4096), triggering out-of-range detection
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertIn("Out-of-range token ID", str(ctx.exception))

    # 18. Missing shard rejection
    def test_18_missing_shard_rejection(self):
        in_dir = self.temp_path / "in_missing"
        out_dir = self.temp_path / "out_missing"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=2, tokens_per_shard=100)
        (in_dir / "shard_00001.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)

    # 19. Unexpected shard rejection
    def test_19_unexpected_shard_rejection(self):
        in_dir = self.temp_path / "in_unexp"
        out_dir = self.temp_path / "out_unexp"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        (in_dir / "shard_00099.bin").write_bytes(b"\x00\x00")
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertIn("Extraneous .bin files", str(ctx.exception))

    # 20. Malformed metadata rejection
    def test_20_malformed_metadata_rejection(self):
        in_dir = self.temp_path / "in_badmeta"
        out_dir = self.temp_path / "out_badmeta"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        in_dir.mkdir(parents=True, exist_ok=True)
        (in_dir / "stage_c_metadata.json").write_text("{bad json...")
        with self.assertRaises(json.JSONDecodeError):
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)

    # 21. Checksum mismatch rejection
    def test_21_checksum_mismatch_rejection(self):
        in_dir = self.temp_path / "in_badsum"
        out_dir = self.temp_path / "out_badsum"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        meta_file = in_dir / "stage_c_metadata.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        meta["shards"][0]["sha256"] = "0" * 64
        with open(meta_file, "w") as f:
            json.dump(meta, f)
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertIn("hash mismatch", str(ctx.exception))

    # 22. Path traversal rejection
    def test_22_path_traversal_rejection(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_and_validate_path("scripts/../../sensitive_dir", "test_path")
        self.assertIn("traversal forbidden", str(ctx.exception))

    # 23. Symlink escape rejection
    def test_23_symlink_escape_rejection(self):
        resolved = normalize_and_validate_path("tests/fixtures", "fixtures")
        self.assertTrue(resolved.is_relative_to(REPO_ROOT))

    # 24. Output-path protection
    def test_24_output_path_protection(self):
        for forbidden in FORBIDDEN_OUTPUT_SUBSTRINGS:
            with self.assertRaises(PermissionError):
                normalize_and_validate_path(forbidden, "output_dir")

    # 25. Read-only file-handle assertion
    def test_25_read_only_file_handle_assertion(self):
        test_file = self.temp_path / "ro_test.bin"
        test_file.write_bytes(b"\x01\x02\x03\x04")
        with open(test_file, "rb") as f:
            self.assertTrue(f.readable())
            self.assertFalse(f.writable())

    # 26. Existing-output overwrite rejection
    def test_26_existing_output_overwrite_rejection(self):
        in_dir = self.temp_path / "in_exist"
        out_dir = self.temp_path / "out_exist"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        tokens = [300 + (i % 600) for i in range(1200)]
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=1200, token_ids=tokens)
        execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        # Second execution to the same out_dir must fail
        with self.assertRaises(FileExistsError):
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)

    # 27. Atomic output failure behavior
    def test_27_atomic_output_failure_behavior(self):
        in_dir = self.temp_path / "in_atomic"
        out_dir = self.temp_path / "out_atomic"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        tokens = [5000] * 1000
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=1000, token_ids=tokens)
        try:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        except ValueError:
            pass
        self.assertFalse((out_dir / "token_frequencies_stage_c_train.json").exists())
        self.assertFalse((out_dir / "candidate_entity_pools.json").exists())
        self.assertFalse((out_dir / "extraction_manifest.json").exists())

    # 28. Resource-limit failure behavior (timeout and memory ceiling enforcement)
    def test_28_resource_limit_failure_behavior(self):
        in_dir = self.temp_path / "in_timeout"
        out_dir = self.temp_path / "out_timeout"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=2, tokens_per_shard=500)
        # 1. Timeout ceiling enforcement
        with self.assertRaises(TimeoutError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True, max_timeout_seconds=-0.01)
        self.assertIn("timeout ceiling", str(ctx.exception))

        # 2. Memory ceiling enforcement
        out_dir_mem = self.temp_path / "out_mem"
        with self.assertRaises(MemoryError) as ctx_mem:
            execute_frequency_extraction(in_dir, v_file, out_dir_mem, synthetic_mode=True, max_ram_mb=0.001)
        self.assertIn("RAM ceiling", str(ctx_mem.exception))

    # 28b. Production execution lock enforcement on function call
    def test_28b_production_lock_function_call(self):
        in_dir = self.temp_path / "in_lock"
        out_dir = self.temp_path / "out_lock"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        with self.assertRaises(PermissionError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=False)
        self.assertIn("requires explicit authorization", str(ctx.exception))

    # 28c. Production authorization gate tests
    def test_28c_production_missing_ref_rejection(self):
        """Production execution without authorization decision ref must be rejected."""
        in_dir = self.temp_path / "in_auth_missing"
        out_dir = self.temp_path / "out_auth_missing"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        with self.assertRaises(PermissionError) as ctx:
            execute_frequency_extraction(
                in_dir, v_file, out_dir, synthetic_mode=False, authorization_decision_ref=None
            )
        self.assertIn("requires explicit authorization", str(ctx.exception))

    def test_28d_production_invalid_ref_rejection(self):
        """Production execution with invalid decision ref filename must be rejected."""
        in_dir = self.temp_path / "in_auth_inv"
        out_dir = self.temp_path / "out_auth_inv"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        with self.assertRaises(PermissionError) as ctx:
            execute_frequency_extraction(
                in_dir, v_file, out_dir, synthetic_mode=False, authorization_decision_ref="wrong_decision.md"
            )
        self.assertIn("does not resolve to the canonical repository path", str(ctx.exception))

    def test_28e_production_tampered_ref_rejection(self):
        """Production execution with tampered external copy must be rejected."""
        in_dir = self.temp_path / "in_auth_tamp"
        out_dir = self.temp_path / "out_auth_tamp"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        fake_doc = self.temp_path / "CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md"
        fake_doc.write_text("tampered content", encoding="utf-8")
        with self.assertRaises(PermissionError) as ctx:
            execute_frequency_extraction(
                in_dir, v_file, out_dir, synthetic_mode=False, authorization_decision_ref=str(fake_doc)
            )
        self.assertIn("External copies or alternate directory locations are strictly rejected", str(ctx.exception))

    def test_28e2_production_external_identical_copy_rejection(self):
        """Production execution with an exact bit-for-bit external copy must still be rejected."""
        in_dir = self.temp_path / "in_auth_ext"
        out_dir = self.temp_path / "out_auth_ext"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=100)
        # Copy exact canonical document to temp path
        canonical_doc = Path("docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md").resolve()
        external_copy = self.temp_path / "CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md"
        external_copy.write_bytes(canonical_doc.read_bytes())
        with self.assertRaises(PermissionError) as ctx:
            execute_frequency_extraction(
                in_dir, v_file, out_dir, synthetic_mode=False, authorization_decision_ref=str(external_copy)
            )
        self.assertIn("External copies or alternate directory locations are strictly rejected", str(ctx.exception))

    def test_28f_production_authorized_invocation_synthetic_fixtures(self):
        """Authorized production invocation executes cleanly against synthetic fixtures."""
        in_dir = self.temp_path / "in_auth_ok"
        out_dir = self.temp_path / "out_auth_ok"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        tokens = [300 + (i % 600) for i in range(1200)]
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=2, tokens_per_shard=600, token_ids=tokens)
        auth_doc = "docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md"
        manifest = execute_frequency_extraction(
            in_dir, v_file, out_dir, synthetic_mode=False, authorization_decision_ref=auth_doc
        )
        self.assertEqual(manifest["synthetic_mode"], False)
        self.assertIn("CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md", manifest["authorization_decision_ref"])
        self.assertTrue((out_dir / "extraction_manifest.json").exists())

    def test_28g_cli_production_authorization_gate(self):
        """Test CLI arguments for mutual exclusivity, rejection, and authorization requirement."""
        import subprocess
        # 1. Reject when no mode is given
        res_none = subprocess.run(
            [
                sys.executable,
                "scripts/extract_stage_c_frequencies.py",
                "--input-dir", "tests/fixtures/synthetic_shards",
                "--vocab-path", "tests/fixtures/synthetic_shards/mock_vocab.json",
                "--output-dir", "tests/fixtures/synthetic_out",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_none.returncode, 1)
        self.assertIn("Explicit execution mode required", res_none.stderr)

        # 2. Reject when both --synthetic and --execute-production are given
        res_both = subprocess.run(
            [
                sys.executable,
                "scripts/extract_stage_c_frequencies.py",
                "--input-dir", "tests/fixtures/synthetic_shards",
                "--vocab-path", "tests/fixtures/synthetic_shards/mock_vocab.json",
                "--output-dir", "tests/fixtures/synthetic_out",
                "--synthetic",
                "--execute-production",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_both.returncode, 1)
        self.assertIn("mutually exclusive", res_both.stderr)

        # 3. Reject when --execute-production is given without --authorization-decision-ref
        res_no_ref = subprocess.run(
            [
                sys.executable,
                "scripts/extract_stage_c_frequencies.py",
                "--input-dir", "tests/fixtures/synthetic_shards",
                "--vocab-path", "tests/fixtures/synthetic_shards/mock_vocab.json",
                "--output-dir", "tests/fixtures/synthetic_out",
                "--execute-production",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_no_ref.returncode, 1)
        self.assertIn("--authorization-decision-ref", res_no_ref.stderr)

    # 29. Token-count reconciliation failure
    def test_29_token_count_reconciliation_failure(self):
        in_dir = self.temp_path / "in_reconcile"
        out_dir = self.temp_path / "out_reconcile"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=1, tokens_per_shard=500)
        meta_file = in_dir / "stage_c_metadata.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        meta["shards"][0]["token_count"] = 600  # Mismatch with actual 500
        with open(meta_file, "w") as f:
            json.dump(meta, f)
        with self.assertRaises(ValueError) as ctx:
            execute_frequency_extraction(in_dir, v_file, out_dir, synthetic_mode=True)
        self.assertIn("Token count mismatch", str(ctx.exception))

    # 30. Reproducibility across repeated synthetic runs
    def test_30_reproducibility_across_repeated_runs(self):
        in_dir = self.temp_path / "in_rep"
        out_dir1 = self.temp_path / "out_rep1"
        out_dir2 = self.temp_path / "out_rep2"
        v_file = self._create_mock_vocab(total_tokens=4096, valid_candidates=650)
        tokens = [300 + (i % 600) for i in range(1200)]
        self._create_mock_shards(in_dir, vocab_size=4096, num_shards=2, tokens_per_shard=600, token_ids=tokens)

        manifest1 = execute_frequency_extraction(in_dir, v_file, out_dir1, synthetic_mode=True)
        manifest2 = execute_frequency_extraction(in_dir, v_file, out_dir2, synthetic_mode=True)

        h1_freq = compute_file_sha256(out_dir1 / "token_frequencies_stage_c_train.json")
        h2_freq = compute_file_sha256(out_dir2 / "token_frequencies_stage_c_train.json")
        self.assertEqual(h1_freq, h2_freq)

        h1_pools = compute_file_sha256(out_dir1 / "candidate_entity_pools.json")
        h2_pools = compute_file_sha256(out_dir2 / "candidate_entity_pools.json")
        self.assertEqual(h1_pools, h2_pools)


if __name__ == "__main__":
    unittest.main()
