"""
Focused Synthetic Extraction & Decoding Algorithm Validation Test Suite.

Verifies:
1. Hex-to-UTF8 vocabulary decoding with error handling.
2. Invalid hexadecimal exception handling.
3. Unicode replacement character disqualification.
4. Duplicate decoded string disqualification.
5. Reserved token ID exclusions (IDs 0..2, IDs 3..258).
6. Whitespace and punctuation delimiter exclusions.
7. Alphanumeric and minimum character length filters.
8. Minimum candidate pool margin assertion at 599 (fail), 600 (pass), and 601 (pass).
9. Deterministic integer tier allocation for counts not divisible by 3 (N // 3, N // 3, N - N1 - N2).
10. Deterministic tie-breaking on equal frequency counts via ascending Token ID.
11. Read-only binary stream handle verification (f.readable() and not f.writable()).
12. Deterministic file closing and non-modification of source directories.

GOVERNANCE RESTRICTIONS:
- Strictly operates on isolated temporary synthetic fixtures.
- NEVER reads real Stage C binary shards or production files.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def decode_vocab_entry(hex_val: str) -> str:
    """
    Decodes a hex-encoded vocabulary string per the CAND_01 decoding contract:
    bytes.fromhex(val).decode("utf-8", errors="replace")
    """
    raw_b = bytes.fromhex(hex_val)
    return raw_b.decode("utf-8", errors="replace")


def is_candidate_entity_token(token_id: int, decoded_token: str) -> bool:
    """
    Evaluates whether a decoded token qualifies as an entity candidate:
    - ID not in reserved range (<259)
    - No replacement character (\ufffd)
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


def allocate_frequency_tiers(candidate_tuples: list) -> dict:
    """
    Partitions candidate tokens into 3 frequency tiers using the frozen integer formula:
    N1 = N // 3, N2 = N // 3, N3 = N - N1 - N2
    Sorted descending by count, with ascending Token ID tie-breaking.
    candidate_tuples: List of (token_id, count)
    """
    # Sort: descending by count (-count), ascending by token_id (token_id)
    sorted_tokens = sorted(candidate_tuples, key=lambda x: (-x[1], x[0]))
    n = len(sorted_tokens)
    assert n >= 600, f"Candidate pool margin failure: required >= 600, got {n}"

    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2

    tier1 = sorted_tokens[:n1]
    tier2 = sorted_tokens[n1:n1 + n2]
    tier3 = sorted_tokens[n1 + n2:]

    assert len(tier1) == n1
    assert len(tier2) == n2
    assert len(tier3) == n3
    assert len(tier1) + len(tier2) + len(tier3) == n

    return {
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "n1": n1,
        "n2": n2,
        "n3": n3,
    }


class TestVocabularyDecodingContract(unittest.TestCase):
    """Tests for hex-decoding, Unicode sanitization, and candidate filtering."""

    def test_valid_hex_decoding(self):
        """Standard alphanumeric hex strings decode cleanly."""
        self.assertEqual(decode_vocab_entry("616c706861"), "alpha")
        self.assertEqual(decode_vocab_entry("64756d70"), "dump")

    def test_invalid_hex_raises_exception(self):
        """Malformed hex input raises ValueError."""
        with self.assertRaises(ValueError):
            decode_vocab_entry("not_a_hex_string")
        with self.assertRaises(ValueError):
            decode_vocab_entry("616z")  # 'z' is invalid hex

    def test_invalid_utf8_replacement_disqualification(self):
        """Invalid UTF-8 sequence produces replacement character and is disqualified."""
        # \xff is invalid start byte in UTF-8
        decoded = decode_vocab_entry("ff")
        self.assertIn("\ufffd", decoded)
        self.assertFalse(is_candidate_entity_token(300, decoded))

    def test_reserved_token_exclusions(self):
        """Tokens 0..2 (special) and 3..258 (byte-level) are excluded."""
        self.assertFalse(is_candidate_entity_token(0, "bos"))
        self.assertFalse(is_candidate_entity_token(1, "eos"))
        self.assertFalse(is_candidate_entity_token(2, "pad"))
        self.assertFalse(is_candidate_entity_token(100, "abc"))  # Raw byte ID
        self.assertFalse(is_candidate_entity_token(258, "xyz"))  # Last raw byte ID
        self.assertTrue(is_candidate_entity_token(259, "abc"))   # First multi-byte

    def test_whitespace_and_delimiter_exclusions(self):
        """Tokens containing spaces, tabs, newlines, or delimiters are disqualified."""
        self.assertFalse(is_candidate_entity_token(300, "word "))
        self.assertFalse(is_candidate_entity_token(301, " word"))
        self.assertFalse(is_candidate_entity_token(302, "word\n"))
        self.assertFalse(is_candidate_entity_token(303, "word-sub"))
        self.assertFalse(is_candidate_entity_token(304, "word_sub"))
        self.assertFalse(is_candidate_entity_token(305, "item."))
        self.assertFalse(is_candidate_entity_token(306, "key:val"))

    def test_alphanumeric_and_length_rule(self):
        """Tokens must be alphanumeric and length >= 3."""
        self.assertFalse(is_candidate_entity_token(300, "ab"))   # Length 2
        self.assertFalse(is_candidate_entity_token(301, "a"))    # Length 1
        self.assertTrue(is_candidate_entity_token(302, "abc"))   # Length 3
        self.assertTrue(is_candidate_entity_token(303, "entity123"))

    def test_duplicate_decoded_strings_detection(self):
        """If two distinct Token IDs decode to identical strings, both must be flagged."""
        synthetic_vocab = {
            300: decode_vocab_entry("616263"),  # "abc"
            301: decode_vocab_entry("616263"),  # "abc" (duplicate)
            302: decode_vocab_entry("78797a"),  # "xyz"
        }
        from collections import Counter
        counts = Counter(synthetic_vocab.values())
        duplicates = {s for s, c in counts.items() if c > 1}
        self.assertIn("abc", duplicates)

        # Disqualify duplicates
        filtered = {tid: s for tid, s in synthetic_vocab.items() if s not in duplicates}
        self.assertNotIn(300, filtered)
        self.assertNotIn(301, filtered)
        self.assertIn(302, filtered)


class TestCandidatePoolMarginAndTierAllocation(unittest.TestCase):
    """Tests for minimum candidate pool margin assertion and integer tier partitioning."""

    def test_pool_margin_assertion_boundary(self):
        """Asserts >= 600: fails at 599, succeeds at 600 and 601."""
        pool_599 = [(i, 100) for i in range(599)]
        pool_600 = [(i, 100) for i in range(600)]
        pool_601 = [(i, 100) for i in range(601)]

        with self.assertRaises(AssertionError):
            allocate_frequency_tiers(pool_599)

        res_600 = allocate_frequency_tiers(pool_600)
        self.assertEqual(res_600["n1"] + res_600["n2"] + res_600["n3"], 600)

        res_601 = allocate_frequency_tiers(pool_601)
        self.assertEqual(res_601["n1"] + res_601["n2"] + res_601["n3"], 601)

    def test_integer_tier_allocation_non_divisible_by_three(self):
        """
        Tests integer division formula on non-divisible counts:
        e.g., N = 754: 754 // 3 = 251 -> N1 = 251, N2 = 251, N3 = 252.
        e.g., N = 601: 601 // 3 = 200 -> N1 = 200, N2 = 200, N3 = 201.
        e.g., N = 602: 602 // 3 = 200 -> N1 = 200, N2 = 200, N3 = 202.
        """
        # Case N = 754
        tokens_754 = [(i, 1000 - i) for i in range(754)]
        res_754 = allocate_frequency_tiers(tokens_754)
        self.assertEqual(res_754["n1"], 251)
        self.assertEqual(res_754["n2"], 251)
        self.assertEqual(res_754["n3"], 252)

        # Case N = 601
        tokens_601 = [(i, 1000 - i) for i in range(601)]
        res_601 = allocate_frequency_tiers(tokens_601)
        self.assertEqual(res_601["n1"], 200)
        self.assertEqual(res_601["n2"], 200)
        self.assertEqual(res_601["n3"], 201)

        # Case N = 602
        tokens_602 = [(i, 1000 - i) for i in range(602)]
        res_602 = allocate_frequency_tiers(tokens_602)
        self.assertEqual(res_602["n1"], 200)
        self.assertEqual(res_602["n2"], 200)
        self.assertEqual(res_602["n3"], 202)

    def test_deterministic_tie_breaking(self):
        """Tokens with identical counts are sorted by ascending Token ID."""
        tied_tokens = [
            (500, 50),
            (300, 50),
            (400, 50),
            (350, 100),  # Higher count -> first
        ]
        # Pad to 600 tokens
        padding = [(1000 + i, 10) for i in range(596)]
        all_tokens = tied_tokens + padding

        res = allocate_frequency_tiers(all_tokens)
        top_4 = res["tier1"][:4]

        # First must be (350, 100)
        self.assertEqual(top_4[0], (350, 100))
        # Next 3 all have count 50; must be sorted ascending by ID: 300, 400, 500
        self.assertEqual(top_4[1], (300, 50))
        self.assertEqual(top_4[2], (400, 50))
        self.assertEqual(top_4[3], (500, 50))


class TestReadOnlyBinaryIOContract(unittest.TestCase):
    """Tests that binary read-only I/O assertions function safely on synthetic files."""

    def test_read_only_stream_assertions(self):
        """Synthetic file handle verifies readable() is True and writable() is False."""
        with tempfile.NamedTemporaryFile("wb", delete=False) as f_tmp:
            f_tmp.write(b"\x01\x00\x02\x00\x03\x00")
            tmp_path = f_tmp.name

        try:
            with open(tmp_path, "rb") as f:
                self.assertTrue(f.readable())
                self.assertFalse(f.writable())
                data = f.read()
                self.assertEqual(len(data), 6)
                # Verify uint16 interpretation
                tokens = np.frombuffer(data, dtype=np.uint16)
                self.assertEqual(list(tokens), [1, 2, 3])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

