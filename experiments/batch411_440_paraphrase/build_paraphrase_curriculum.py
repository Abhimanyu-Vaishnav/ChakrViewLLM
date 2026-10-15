"""
Batches 416-422: Comprehensive Multi-Domain Paraphrase & Generalization Dataset Builder.
Location: experiments/batch411_440_paraphrase/build_paraphrase_curriculum.py

Creates strictly partitioned manifests:
1. train_augmented.jsonl: Training split with 4-5 diverse phrasing variants per concept.
2. test_heldout_paraphrase.jsonl: Strictly held-out paraphrase variants of learned concepts.
3. test_heldout_compositional.jsonl: Novel compositions and multi-step tasks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List

DATA_DIR = Path("experiments/batch411_440_paraphrase/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Core Concept Families with 5-6 diverse phrasing variants per concept
CONCEPT_FAMILIES = [
    # Family 1: Arithmetic Addition (14 + 28 = 42)
    {
        "concept_id": "math_add_14_28",
        "domain": "mathematics",
        "target": " 42",
        "expected": "42",
        "train_variants": [
            "Question: What is 14 plus 28?\nAnswer:",
            "Calculate the sum of 14 and 28.\nAnswer:",
            "Compute 14 + 28:\nResult:",
            "If you add twenty-eight to fourteen, what is the total?\nAnswer:",
        ],
        "heldout_variants": [
            "Compute the sum of fourteen and twenty eight.\nAnswer:",
            "Find the total value when 28 is added to 14:\nValue:",
        ],
    },
    # Family 2: Arithmetic Subtraction (50 - 18 = 32)
    {
        "concept_id": "math_sub_50_18",
        "domain": "mathematics",
        "target": " 32",
        "expected": "32",
        "train_variants": [
            "Question: What is 50 minus 18?\nAnswer:",
            "Calculate 50 subtracted by 18.\nAnswer:",
            "Compute 50 - 18:\nResult:",
            "Subtract eighteen from fifty.\nAnswer:",
        ],
        "heldout_variants": [
            "What remains when you take 18 away from 50?\nAnswer:",
            "Evaluate the difference between fifty and eighteen:\nDifference:",
        ],
    },
    # Family 3: Arithmetic Multiplication (7 * 6 = 42)
    {
        "concept_id": "math_mul_7_6",
        "domain": "mathematics",
        "target": " 42",
        "expected": "42",
        "train_variants": [
            "Question: What is 7 multiplied by 6?\nAnswer:",
            "Calculate the product of 7 and 6.\nAnswer:",
            "Compute 7 * 6:\nResult:",
            "What is seven times six?\nAnswer:",
        ],
        "heldout_variants": [
            "Find the multiplication of seven and six.\nAnswer:",
            "What is the mathematical product of 6 and 7?\nProduct:",
        ],
    },
    # Family 4: Computer Architecture - ALU
    {
        "concept_id": "arch_alu",
        "domain": "computer_architecture",
        "target": " Arithmetic Logic Unit (ALU)",
        "expected": "Arithmetic Logic Unit (ALU)",
        "train_variants": [
            "Question: What component of the CPU executes arithmetic and logic operations?\nAnswer:",
            "Which processor block performs binary math and bitwise calculations?\nAnswer:",
            "Identify the CPU subsystem responsible for computing integer arithmetic and logic.\nAnswer:",
            "In microarchitecture, what unit evaluates additions, subtractions, and bitwise AND/OR?\nAnswer:",
        ],
        "heldout_variants": [
            "What circuit in a CPU combines arithmetic operations with logic gates?\nAnswer:",
            "Name the primary compute hardware engine located inside a central processor core.\nAnswer:",
        ],
    },
    # Family 5: Computer Architecture - DMA
    {
        "concept_id": "arch_dma",
        "domain": "computer_architecture",
        "target": " Direct Memory Access",
        "expected": "Direct Memory Access",
        "train_variants": [
            "Question: What does DMA stand for in computer systems?\nAnswer:",
            "Expand the abbreviation DMA in computer engineering.\nAnswer:",
            "In computing architecture, what does the acronym DMA represent?\nAnswer:",
            "What is the full form of DMA used for memory-peripheral transfers?\nAnswer:",
        ],
        "heldout_variants": [
            "In microprocessor design, what does the abbreviation DMA represent?\nAnswer:",
            "What hardware feature allows I/O devices to bypass the CPU when writing to RAM?\nAnswer:",
        ],
    },
    # Family 6: Python Coding - Square Function
    {
        "concept_id": "code_square",
        "domain": "coding",
        "target": "def square(x):\n    return x * x",
        "expected": "def square(x):\n    return x * x",
        "train_variants": [
            "Write a Python function to compute the square of x.\nCode:\n",
            "Define a function square(x) in Python that returns x multiplied by itself.\nCode:\n",
            "Implement a Python helper returning the square of a parameter x.\nCode:\n",
            "Write Python code for def square(x) computing x squared.\nCode:\n",
        ],
        "heldout_variants": [
            "Create a Python function named square that squares its input argument x.\nCode:\n",
            "Write a clean Python implementation that computes x raised to the power of two.\nCode:\n",
        ],
    },
    # Family 7: Python Coding - Is Even Function
    {
        "concept_id": "code_is_even",
        "domain": "coding",
        "target": "def is_even(n):\n    return n % 2 == 0",
        "expected": "def is_even(n):\n    return n % 2 == 0",
        "train_variants": [
            "Write a Python function is_even(n) that returns True if n is even.\nCode:\n",
            "Define an is_even(n) function in Python checking if integer n has zero remainder when divided by 2.\nCode:\n",
            "Implement is_even(n) in Python using modulo operator.\nCode:\n",
            "Write a Python predicate returning True when number n is divisible by two.\nCode:\n",
        ],
        "heldout_variants": [
            "Write a Python function to test whether an integer n is an even number.\nCode:\n",
            "Define def is_even(n) returning boolean evaluation of n modulo 2 equality to zero.\nCode:\n",
        ],
    },
    # Family 8: Science - Triangle Angle Sum
    {
        "concept_id": "sci_triangle",
        "domain": "science",
        "target": " 180",
        "expected": "180",
        "train_variants": [
            "Question: What is the sum of interior angles in a triangle in degrees?\nAnswer:",
            "How many degrees do the three angles inside a triangle add up to?\nAnswer:",
            "In Euclidean geometry, what is the total angle sum of any triangle?\nAnswer:",
            "State the degree sum of a triangle's internal angles.\nAnswer:",
        ],
        "heldout_variants": [
            "What is the total degree measurement when you sum all three interior angles of a triangle?\nAnswer:",
            "Calculate the interior angle summation for a planar triangle in degrees.\nAnswer:",
        ],
    },
    # Family 9: Multilingual (Hindi) - Capital of India
    {
        "concept_id": "hindi_capital",
        "domain": "hindi",
        "target": " नई दिल्ली",
        "expected": "नई दिल्ली",
        "train_variants": [
            "प्रश्न: भारत की राजधानी क्या है?\nउत्तर:",
            "भारत देश की राष्ट्रीय राजधानी का नाम क्या है?\nउत्तर:",
            "भारतीय संघ की राजधानी कौन सा महानगर है?\nउत्तर:",
            "भारत की प्रशासनिक राजधानी का नाम बताएं।\nउत्तर:",
        ],
        "heldout_variants": [
            "भारतीय गणराज्य की राष्ट्रीय राजधानी कौन सा शहर है?\nउत्तर:",
            "भारत देश का शासन केंद्र और राजधानी क्या कहलाती है?\nउत्तर:",
        ],
    },
    # Family 10: Multilingual (Hindi) - Sun Rises in the East
    {
        "concept_id": "hindi_sun_east",
        "domain": "hindi",
        "target": " पूर्व",
        "expected": "पूर्व",
        "train_variants": [
            "प्रश्न: सूर्य किस दिशा से उगता है?\nउत्तर:",
            "सूरज के उदय होने की दिशा कौन सी है?\nउत्तर:",
            "सूर्य प्रातः काल किस दिशा में दिखाई देता है?\nउत्तर:",
            "सूर्योदय किस मुख्य दिशा में होता है?\nउत्तर:",
        ],
        "heldout_variants": [
            "पृथ्वी पर सूर्य किस दिशा की ओर से निकलता है?\nउत्तर:",
            "सूर्य के निकलने की भौगोलिक दिशा बताइए।\nउत्तर:",
        ],
    },
]

# Separate Compositional Test Manifest
COMPOSITIONAL_ITEMS = [
    {
        "id": "comp_001",
        "domain": "compositional",
        "query": "Calculate the square of (3 plus 5).",
        "prompt": "Question: Calculate the square of (3 plus 5).\nAnswer:",
        "target": " 64",
        "expected": "64",
    },
    {
        "id": "comp_002",
        "domain": "compositional",
        "query": "What is 7 multiplied by 6 minus 10?",
        "prompt": "Question: What is 7 multiplied by 6 minus 10?\nAnswer:",
        "target": " 32",
        "expected": "32",
    },
    {
        "id": "comp_003",
        "domain": "compositional",
        "query": "Calculate (14 plus 28) minus 10.",
        "prompt": "Question: Calculate (14 plus 28) minus 10.\nAnswer:",
        "target": " 32",
        "expected": "32",
    },
    {
        "id": "comp_004",
        "domain": "compositional",
        "query": "In a triangle, if two angles are 60 and 70 degrees, what is the third angle?",
        "prompt": "Question: In a triangle, if two angles are 60 and 70 degrees, what is the third angle?\nAnswer:",
        "target": " 50",
        "expected": "50",
    },
]


def build_and_export():
    train_records = []
    heldout_records = []

    train_uid = 0
    heldout_uid = 0

    for fam in CONCEPT_FAMILIES:
        cid = fam["concept_id"]
        dom = fam["domain"]
        target = fam["target"]
        expected = fam["expected"]

        # Training variants
        for prompt in fam["train_variants"]:
            train_uid += 1
            train_records.append({
                "id": f"train_{cid}_{train_uid:03d}",
                "concept_id": cid,
                "domain": dom,
                "prompt": prompt,
                "target": target,
                "expected": expected,
            })

        # Held-out paraphrase variants
        for prompt in fam["heldout_variants"]:
            heldout_uid += 1
            heldout_records.append({
                "id": f"heldout_{cid}_{heldout_uid:03d}",
                "concept_id": cid,
                "domain": dom,
                "prompt": prompt,
                "target": target,
                "expected": expected,
            })

    # Export Training Manifest
    train_file = DATA_DIR / "train_augmented.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Export Heldout Paraphrase Manifest
    heldout_file = DATA_DIR / "test_heldout_paraphrase.jsonl"
    with open(heldout_file, "w", encoding="utf-8") as f:
        for r in heldout_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Export Compositional Manifest
    comp_file = DATA_DIR / "test_heldout_compositional.jsonl"
    with open(comp_file, "w", encoding="utf-8") as f:
        for r in COMPOSITIONAL_ITEMS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def file_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            while chunk := fp.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    print(f"Generated Training Manifest: {len(train_records)} items at {train_file} (SHA256: {file_hash(train_file)})")
    print(f"Generated Heldout Paraphrase: {len(heldout_records)} items at {heldout_file} (SHA256: {file_hash(heldout_file)})")
    print(f"Generated Heldout Compositional: {len(COMPOSITIONAL_ITEMS)} items at {comp_file} (SHA256: {file_hash(comp_file)})")


if __name__ == "__main__":
    build_and_export()
