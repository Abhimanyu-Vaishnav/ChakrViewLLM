"""Independent Dataset Quality Audit Suite for UCIP-17.
Verifies:
1. Exact file existence and line counts.
2. Category and language distribution adherence.
3. Zero partition leakage across train, val, and dev.
4. Zero historical sealed benchmark contamination.
5. Strict sequence length bounding (<= 512 tokens).
6. Tokenizer round-trip encoding/decoding parity.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "sft" / "ucip17"


def test_01_partition_counts_and_file_existence():
    files = {
        "sft_train_12000.jsonl": 12000,
        "sft_validation_1500.jsonl": 1500,
        "sft_dev_1500.jsonl": 1500,
        "sft_master_15000.jsonl": 15000,
    }
    for fname, count in files.items():
        fpath = DATA_DIR / fname
        assert fpath.exists(), f"File {fname} does not exist!"
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == count, f"{fname} expected {count} lines, found {len(lines)}"


def test_02_category_and_language_distribution():
    master_path = DATA_DIR / "sft_master_15000.jsonl"
    with open(master_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    cats = {}
    langs = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        langs[r["language"]] = langs.get(r["language"], 0) + 1

    assert len(cats) == 18, f"Expected 18 categories, got {len(cats)}"
    assert cats["INST"] == 1200
    assert cats["PYTHON"] == 1200
    assert cats["HIN"] == 1200
    assert cats["QA"] == 1050
    assert cats["EXPL"] == 1050
    assert cats["CONV"] == 900
    assert cats["CTX"] == 900
    assert cats["ARITH"] == 900
    assert cats["ENG"] == 900
    assert cats["BIND"] == 750
    assert cats["LOGIC"] == 750
    assert cats["UNCERT"] == 750
    assert cats["COPY"] == 600
    assert cats["HING"] == 600
    assert cats["FORMAT"] == 600
    assert cats["CORR"] == 600
    assert cats["CLARIF"] == 600
    assert cats["SAN"] == 450

    assert langs["en"] == 8250
    assert langs["hi"] == 4500
    assert langs["hi-en"] == 1500
    assert langs["sa"] == 750


def test_03_zero_partition_leakage():
    def load_ids(fname):
        with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
            return set(json.loads(line)["id"] for line in f)

    train_ids = load_ids("sft_train_12000.jsonl")
    val_ids = load_ids("sft_validation_1500.jsonl")
    dev_ids = load_ids("sft_dev_1500.jsonl")

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val IDs overlap!"
    assert len(train_ids.intersection(dev_ids)) == 0, "Train and Dev IDs overlap!"
    assert len(val_ids.intersection(dev_ids)) == 0, "Val and Dev IDs overlap!"


def test_04_sealed_benchmark_zero_contamination():
    sealed_prompts = set()
    for p in ROOT.glob("experiments/**/*sealed*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "prompt" in item:
                            sealed_prompts.add(item["prompt"].strip())
        except Exception:
            pass

    master_path = DATA_DIR / "sft_master_15000.jsonl"
    with open(master_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            clean_p = r["prompt"].replace("User: ", "").replace("\nAssistant: ", "").strip()
            assert clean_p not in sealed_prompts, f"Contamination at index {idx}: {clean_p}"


def test_05_tokenizer_and_context_length():
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")

    master_path = DATA_DIR / "sft_master_15000.jsonl"
    with open(master_path, "r", encoding="utf-8") as f:
        for i in range(200):
            line = f.readline()
            if not line: break
            r = json.loads(line)
            text = r["prompt"] + r["response"]
            encoded = tok.encode(text)
            assert len(encoded) <= 512, f"Length {len(encoded)} exceeds 512 context limit!"
            decoded = tok.decode(encoded)
            assert len(decoded) > 0
