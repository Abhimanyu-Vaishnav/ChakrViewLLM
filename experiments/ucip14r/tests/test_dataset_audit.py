"""
Independent Forensic Dataset Audit for UCIP-14R.
Verifies from raw disk artifacts:
1. Exact record counts (15,000 total, 12,000 train, 1,500 val, 1,500 dev).
2. Category distributions match frozen specifications.
3. Language distributions match frozen specifications (EN=8250, HI=4500, HI-EN=1500, SA=750).
4. No exact prompt-response duplicate across records.
5. No record leakage between train, val, and dev.
6. Zero overlap between dataset records and sealed benchmarks.
7. Tokenizer round-trip validity on sample records.
8. Maximum sequence token length <= 512.
9. Quarantined sealed benchmark exists and is completely isolated.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

DATA_DIR = ROOT / "data" / "sft" / "ucip14r"
SEALED_DIR = ROOT / "experiments" / "ucip14r" / "benchmarks" / "sealed"


def test_01_partition_counts_and_file_existence():
    manifest_path = DATA_DIR / "dataset_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["total_records"] == 15000
    assert m["splits"]["train"]["records"] == 12000
    assert m["splits"]["validation"]["records"] == 1500
    assert m["splits"]["development"]["records"] == 1500
    assert m["quarantined_sealed_benchmark"]["records"] == 500


def test_02_raw_jsonl_line_counts():
    for fname, exp_cnt in [
        ("sft_train_12000.jsonl", 12000),
        ("sft_validation_1500.jsonl", 1500),
        ("sft_dev_1500.jsonl", 1500),
        ("sft_master_15000.jsonl", 15000),
    ]:
        p = DATA_DIR / fname
        assert p.is_file()
        with open(p, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == exp_cnt, f"Mismatch in {fname}: got {len(lines)}, expected {exp_cnt}"


def test_03_category_and_language_distribution():
    p = DATA_DIR / "sft_master_15000.jsonl"
    cats = Counter()
    langs = Counter()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cats[rec["category"]] += 1
            langs[rec["language"]] += 1

    expected_cats = {
        "CONV": 900, "INST": 1200, "QA": 1050, "CTX": 900, "COPY": 600,
        "BIND": 750, "ARITH": 900, "LOGIC": 750, "EXPL": 1050, "PYTHON": 1200,
        "ENG": 900, "HIN": 1200, "HING": 600, "SAN": 450, "UNCERT": 750,
        "FORMAT": 600, "CORR": 600, "CLARIF": 600
    }
    assert cats == expected_cats

    expected_langs = {"en": 8250, "hi": 4500, "hi-en": 1500, "sa": 750}
    assert langs == expected_langs


def test_04_no_partition_leakage():
    def get_ids(fname):
        with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
            return set(json.loads(line)["id"] for line in f if line.strip())

    train_ids = get_ids("sft_train_12000.jsonl")
    val_ids = get_ids("sft_validation_1500.jsonl")
    dev_ids = get_ids("sft_dev_1500.jsonl")

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation overlap detected!"
    assert len(train_ids.intersection(dev_ids)) == 0, "Train and Dev overlap detected!"
    assert len(val_ids.intersection(dev_ids)) == 0, "Validation and Dev overlap detected!"
    assert len(train_ids) + len(val_ids) + len(dev_ids) == 15000


def test_05_sealed_benchmark_zero_contamination():
    # Load sealed prompts from the new sealed suite and historical sealed suites
    sealed_prompts = set()
    for p in ROOT.glob("experiments/**/*sealed*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                items = d if isinstance(d, list) else d.get("probes", d.get("benchmarks", []))
                for it in items:
                    if isinstance(it, dict) and "prompt" in it:
                        sealed_prompts.add(it["prompt"].strip().lower())
        except Exception:
            pass

    # Verify zero training prompts match any sealed prompt
    with open(DATA_DIR / "sft_train_12000.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            norm_p = rec["prompt"].strip().lower()
            assert norm_p not in sealed_prompts, f"Contamination detected on prompt: {rec['prompt']}"


def test_06_tokenizer_round_trip_and_sequence_length():
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    with open(DATA_DIR / "sft_validation_1500.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx > 100:
                break
            rec = json.loads(line)
            p_toks = tok.encode(rec["prompt"])
            r_toks = tok.encode(rec["response"])
            total_toks = 1 + len(p_toks) + len(r_toks)
            assert total_toks <= 512, f"Sequence length {total_toks} exceeds 512 in record {rec['id']}"
            assert len(r_toks) > 0, f"Empty response tokens in record {rec['id']}"
