"""
Batches 386-393: Clean Diagnostic Benchmark & Multi-Domain Evaluation Set.
Location: experiments/batch381_410_integrity/data/diagnostic_answer_set.jsonl

Categories:
1. Elementary Arithmetic (Addition, Subtraction, Multiplication).
2. Computer Architecture & Systems (CPU, ALU, Pipeline, Memory Hierarchy).
3. Python Coding (Function signatures, return expressions).
4. Basic Science & Geometry (Angles in triangle, gravity, solar system).
5. Hindi / Multilingual (Geography, definitions).
6. Compositional Two-Step Math & Reasoning.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_FILE = Path("experiments/batch381_410_integrity/data/diagnostic_answer_set.jsonl")


def build_dataset():
    records = [
        # Domain 1: Arithmetic & Simple Word Problems
        {
            "id": "math_001",
            "domain": "mathematics",
            "capability_level": "familiar_pattern",
            "query": "What is 14 plus 28?",
            "prompt": "Question: What is 14 plus 28?\nAnswer:",
            "target": " 42",
            "expected_answer": "42",
        },
        {
            "id": "math_002",
            "domain": "mathematics",
            "capability_level": "familiar_pattern",
            "query": "What is 50 minus 18?",
            "prompt": "Question: What is 50 minus 18?\nAnswer:",
            "target": " 32",
            "expected_answer": "32",
        },
        {
            "id": "math_003",
            "domain": "mathematics",
            "capability_level": "familiar_pattern",
            "query": "What is 7 multiplied by 6?",
            "prompt": "Question: What is 7 multiplied by 6?\nAnswer:",
            "target": " 42",
            "expected_answer": "42",
        },
        {
            "id": "math_004",
            "domain": "mathematics",
            "capability_level": "compositional",
            "query": "Calculate (10 plus 5) multiplied by 2.",
            "prompt": "Question: Calculate (10 plus 5) multiplied by 2.\nAnswer:",
            "target": " 30",
            "expected_answer": "30",
        },

        # Domain 2: Computer Architecture & Systems
        {
            "id": "arch_001",
            "domain": "computer_architecture",
            "capability_level": "memorized_concept",
            "query": "What component of the CPU executes arithmetic and logic operations?",
            "prompt": "Question: What component of the CPU executes arithmetic and logic operations?\nAnswer:",
            "target": " Arithmetic Logic Unit (ALU)",
            "expected_answer": "Arithmetic Logic Unit (ALU)",
        },
        {
            "id": "arch_002",
            "domain": "computer_architecture",
            "capability_level": "memorized_concept",
            "query": "What does DMA stand for in computer systems?",
            "prompt": "Question: What does DMA stand for in computer systems?\nAnswer:",
            "target": " Direct Memory Access",
            "expected_answer": "Direct Memory Access",
        },
        {
            "id": "arch_003",
            "domain": "computer_architecture",
            "capability_level": "familiar_pattern",
            "query": "What memory level sits between the L1 cache and main RAM?",
            "prompt": "Question: What memory level sits between the L1 cache and main RAM?\nAnswer:",
            "target": " L2 and L3 Cache",
            "expected_answer": "L2 and L3 Cache",
        },

        # Domain 3: Python Coding
        {
            "id": "code_001",
            "domain": "coding",
            "capability_level": "familiar_pattern",
            "query": "Write a Python function to compute the square of x.",
            "prompt": "Write a Python function to compute the square of x.\nCode:\n",
            "target": "def square(x):\n    return x * x",
            "expected_answer": "def square(x):\n    return x * x",
        },
        {
            "id": "code_002",
            "domain": "coding",
            "capability_level": "familiar_pattern",
            "query": "Write a Python function is_even(n) that returns True if n is even.",
            "prompt": "Write a Python function is_even(n) that returns True if n is even.\nCode:\n",
            "target": "def is_even(n):\n    return n % 2 == 0",
            "expected_answer": "def is_even(n):\n    return n % 2 == 0",
        },

        # Domain 4: Basic Science & Geometry
        {
            "id": "sci_001",
            "domain": "science",
            "capability_level": "familiar_pattern",
            "query": "What is the sum of interior angles in a triangle in degrees?",
            "prompt": "Question: What is the sum of interior angles in a triangle in degrees?\nAnswer:",
            "target": " 180",
            "expected_answer": "180",
        },
        {
            "id": "sci_002",
            "domain": "science",
            "capability_level": "familiar_pattern",
            "query": "Who formulated the universal law of gravitation?",
            "prompt": "Question: Who formulated the universal law of gravitation?\nAnswer:",
            "target": " Sir Isaac Newton",
            "expected_answer": "Sir Isaac Newton",
        },

        # Domain 5: Multilingual (Hindi)
        {
            "id": "hindi_001",
            "domain": "hindi",
            "capability_level": "familiar_pattern",
            "query": "सूर्य किस दिशा से उगता है?",
            "prompt": "प्रश्न: सूर्य किस दिशा से उगता है?\nउत्तर:",
            "target": " पूर्व",
            "expected_answer": "पूर्व",
        },
        {
            "id": "hindi_002",
            "domain": "hindi",
            "capability_level": "familiar_pattern",
            "query": "भारत की राजधानी क्या है?",
            "prompt": "प्रश्न: भारत की राजधानी क्या है?\nउत्तर:",
            "target": " नई दिल्ली",
            "expected_answer": "नई दिल्ली",
        },

        # Domain 6: Unseen Compositional Transfer
        {
            "id": "unseen_001",
            "domain": "unseen_composition",
            "capability_level": "unseen_composition",
            "query": "What memory component is used by Neuromorphic hardware to emulate biological synapses?",
            "prompt": "Question: What memory component is used by Neuromorphic hardware to emulate biological synapses?\nAnswer:",
            "target": " Memristor",
            "expected_answer": "Memristor",
        },
        {
            "id": "unseen_002",
            "domain": "unseen_composition",
            "capability_level": "unseen_composition",
            "query": "Calculate the square of (3 plus 5).",
            "prompt": "Question: Calculate the square of (3 plus 5).\nAnswer:",
            "target": " 64",
            "expected_answer": "64",
        },
    ]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} diagnostic examples at {OUT_FILE}")


if __name__ == "__main__":
    build_dataset()
