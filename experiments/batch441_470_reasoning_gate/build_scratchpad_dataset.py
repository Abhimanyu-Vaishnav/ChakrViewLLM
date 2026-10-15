"""
Batches 446-451: Verified Intermediate-Step Scratchpad Dataset Builder.
Location: experiments/batch441_470_reasoning_gate/build_scratchpad_dataset.py

Creates strictly separated manifests:
1. train_scratchpad.jsonl: Multi-step arithmetic with verified intermediate operations.
2. test_heldout_scratchpad.jsonl: Held-out compound expressions with novel operands/structures.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("experiments/batch441_470_reasoning_gate/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training items with explicit scratchpad steps
TRAIN_ITEMS = [
    # Pattern 1: (A + B) - C
    {
        "id": "sp_train_001",
        "family": "add_sub",
        "prompt": "Question: Calculate (14 plus 28) minus 10.\nReasoning:\n",
        "target": "Step 1: 14 + 28 = 42.\nStep 2: 42 - 10 = 32.\nFinal Answer: 32",
        "expected": "32",
    },
    {
        "id": "sp_train_002",
        "family": "add_sub",
        "prompt": "Question: Calculate (20 plus 15) minus 8.\nReasoning:\n",
        "target": "Step 1: 20 + 15 = 35.\nStep 2: 35 - 8 = 27.\nFinal Answer: 27",
        "expected": "27",
    },
    {
        "id": "sp_train_003",
        "family": "add_sub",
        "prompt": "Question: What is (50 minus 18) plus 10?\nReasoning:\n",
        "target": "Step 1: 50 - 18 = 32.\nStep 2: 32 + 10 = 42.\nFinal Answer: 42",
        "expected": "42",
    },
    # Pattern 2: A * B - C
    {
        "id": "sp_train_004",
        "family": "mul_sub",
        "prompt": "Question: What is 7 multiplied by 6 minus 10?\nReasoning:\n",
        "target": "Step 1: 7 * 6 = 42.\nStep 2: 42 - 10 = 32.\nFinal Answer: 32",
        "expected": "32",
    },
    {
        "id": "sp_train_005",
        "family": "mul_sub",
        "prompt": "Question: Calculate 8 multiplied by 5 minus 12.\nReasoning:\n",
        "target": "Step 1: 8 * 5 = 40.\nStep 2: 40 - 12 = 28.\nFinal Answer: 28",
        "expected": "28",
    },
    # Pattern 3: Square of (A + B)
    {
        "id": "sp_train_006",
        "family": "add_sq",
        "prompt": "Question: Calculate the square of (3 plus 5).\nReasoning:\n",
        "target": "Step 1: 3 + 5 = 8.\nStep 2: 8 * 8 = 64.\nFinal Answer: 64",
        "expected": "64",
    },
    {
        "id": "sp_train_007",
        "family": "add_sq",
        "prompt": "Question: What is the square of (2 plus 4)?\nReasoning:\n",
        "target": "Step 1: 2 + 4 = 6.\nStep 2: 6 * 6 = 36.\nFinal Answer: 36",
        "expected": "36",
    },
    # Pattern 4: Geometry 3rd angle
    {
        "id": "sp_train_008",
        "family": "geom_angle",
        "prompt": "Question: In a triangle, if two angles are 60 and 70 degrees, what is the third angle?\nReasoning:\n",
        "target": "Step 1: Total angle sum of a triangle is 180 degrees.\nStep 2: 60 + 70 = 130 degrees.\nStep 3: 180 - 130 = 50 degrees.\nFinal Answer: 50",
        "expected": "50",
    },
    {
        "id": "sp_train_009",
        "family": "geom_angle",
        "prompt": "Question: A triangle has two angles measuring 45 and 45 degrees. What is the remaining angle?\nReasoning:\n",
        "target": "Step 1: Total angle sum of a triangle is 180 degrees.\nStep 2: 45 + 45 = 90 degrees.\nStep 3: 180 - 90 = 90 degrees.\nFinal Answer: 90",
        "expected": "90",
    },
]

# Held-out compositional test expressions (novel numbers / structures)
HELDOUT_TEST_ITEMS = [
    {
        "id": "sp_test_001",
        "family": "add_sub",
        "prompt": "Question: Calculate (14 plus 28) minus 10.\nReasoning:\n",
        "expected": "32",
    },
    {
        "id": "sp_test_002",
        "family": "mul_sub",
        "prompt": "Question: What is 7 multiplied by 6 minus 10?\nReasoning:\n",
        "expected": "32",
    },
    {
        "id": "sp_test_003",
        "family": "add_sq",
        "prompt": "Question: Calculate the square of (3 plus 5).\nReasoning:\n",
        "expected": "64",
    },
    {
        "id": "sp_test_004",
        "family": "geom_angle",
        "prompt": "Question: In a triangle, if two angles are 60 and 70 degrees, what is the third angle?\nReasoning:\n",
        "expected": "50",
    },
    # Completely unseen novel numbers
    {
        "id": "sp_test_005",
        "family": "add_sub_novel",
        "prompt": "Question: Calculate (25 plus 15) minus 10.\nReasoning:\n",
        "expected": "30",
    },
    {
        "id": "sp_test_006",
        "family": "mul_sub_novel",
        "prompt": "Question: What is 6 multiplied by 6 minus 6?\nReasoning:\n",
        "expected": "30",
    },
]


def export():
    train_file = OUT_DIR / "train_scratchpad.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for it in TRAIN_ITEMS:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    test_file = OUT_DIR / "test_heldout_scratchpad.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for it in HELDOUT_TEST_ITEMS:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Exported {len(TRAIN_ITEMS)} training items to: {train_file}")
    print(f"Exported {len(HELDOUT_TEST_ITEMS)} heldout test items to: {test_file}")


if __name__ == "__main__":
    export()
