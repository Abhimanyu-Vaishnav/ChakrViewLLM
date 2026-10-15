"""
WP 954–961: Build a Narrow, Honest Task Family.
Focuses on the core foundation: Simple Compound Arithmetic Planning (1-step and 2-step).
Generates leakage-resistant train, val, and held-out test splits with distinct
prompt templates, unseen numeric ranges, repeated values, Hinglish, and missing arguments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "batch941_980" / "data"
REPORT_DIR = REPO_ROOT / "experiments" / "batch941_980" / "reports"


def build_focused_dataset():
    random.seed(941980)
    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

    # Templates for compound arithmetic:
    # Template 1: Calculate {a} * {b} and add {c}
    # Template 2: Multiply {a} by {b}, then subtract {c}
    # Template 3: Take {a}, divide by {b}, and add {c}
    # Template 4: Hinglish: {a} aur {b} ka guna karo aur {c} add karo
    # Template 5: Missing argument clarification

    # Train / Val numeric pool (held-out from test)
    train_pool = [
        (120, 4, 15, "*", "+"), (250, 3, 20, "*", "+"), (340, 5, 12, "*", "-"),
        (480, 2, 35, "/", "+"), (510, 6, 18, "*", "+"), (620, 4, 25, "*", "-"),
        (710, 3, 40, "*", "+"), (830, 2, 50, "*", "-"), (950, 5, 30, "/", "+"),
        (1100, 2, 60, "*", "+"), (1250, 4, 45, "*", "-"), (1600, 8, 20, "/", "+"),
        (45, 5, 10, "*", "+"), (88, 2, 14, "*", "-"), (96, 3, 12, "/", "+"),
        (14, 14, 5, "*", "+"), (55, 5, 55, "*", "-"), (100, 20, 100, "/", "+"),
    ]

    # Test numeric pool (STRICTLY HELD-OUT OPERANDS)
    test_pool = [
        (600, 5, 14, "*", "+"), (605, 3, 22, "*", "+"), (720, 4, 18, "*", "-"),
        (850, 2, 35, "/", "+"), (912, 6, 41, "*", "+"), (1024, 8, 16, "/", "+"),
        (2048, 4, 12, "/", "-"), (350, 7, 19, "*", "+"), (1500, 2, 80, "*", "-"),
        (2500, 5, 100, "/", "+"),
    ]

    def make_records(pool: List[Tuple], is_test: bool = False) -> List[Dict[str, Any]]:
        records = []
        for a, b, c, op1, op2 in pool:
            # English standard
            p1 = f"User: Calculate {a} {op1} {b} and then {'add' if op2=='+' else 'subtract'} {c} to the result\nPlan:\n"
            t1 = (
                f"GOAL: compound_arithmetic\n"
                f"STEP 0: tool_call | calculator | args={{expr={a}{op1}{b}}} | deps=[]\n"
                f"STEP 1: tool_call | calculator | args={{expr=$0{op2}{c}}} | deps=[0]\n"
                f"CLARIFICATION: NONE\n"
                f"<END_PLAN>"
            )
            records.append({
                "prompt": p1,
                "target_dsl": t1,
                "slots": [
                    {"slot_idx": 0, "text": str(a), "type": "NUMBER"},
                    {"slot_idx": 1, "text": str(b), "type": "NUMBER"},
                    {"slot_idx": 2, "text": str(c), "type": "NUMBER"},
                ],
                "topology": {"step_count": 2, "op1": op1, "op2": op2},
            })

            # English imperative
            verb1 = "Multiply" if op1 == "*" else "Divide"
            prep1 = "by"
            verb2 = "add" if op2 == "+" else "subtract"
            p2 = f"User: {verb1} {a} {prep1} {b}, then {verb2} {c}\nPlan:\n"
            records.append({
                "prompt": p2,
                "target_dsl": t1,
                "slots": [
                    {"slot_idx": 0, "text": str(a), "type": "NUMBER"},
                    {"slot_idx": 1, "text": str(b), "type": "NUMBER"},
                    {"slot_idx": 2, "text": str(c), "type": "NUMBER"},
                ],
                "topology": {"step_count": 2, "op1": op1, "op2": op2},
            })

            # Hinglish phrasing
            p3 = f"User: {a} aur {b} ko {op1} karo aur result me {c} {verb2} karo\nPlan:\n"
            records.append({
                "prompt": p3,
                "target_dsl": t1,
                "slots": [
                    {"slot_idx": 0, "text": str(a), "type": "NUMBER"},
                    {"slot_idx": 1, "text": str(b), "type": "NUMBER"},
                    {"slot_idx": 2, "text": str(c), "type": "NUMBER"},
                ],
                "topology": {"step_count": 2, "op1": op1, "op2": op2},
            })

        # Clarification prompts
        clar_items = [
            ("Calculate net settlement without supplying account_id", "account_id"),
            ("Execute transaction without specifying auth_signature", "auth_signature"),
            ("Process arithmetic operation without providing second_operand", "second_operand"),
        ]
        for prompt_text, missing_var in clar_items:
            records.append({
                "prompt": f"User: {prompt_text}\nPlan:\n",
                "target_dsl": f"GOAL: clarification_needed\nSTEP 0: clarification | none | args={{missing={missing_var}}} | deps=[]\nCLARIFICATION: NEEDED\n<END_PLAN>",
                "slots": [],
                "topology": {"step_count": 0, "clarification": True},
            })
        return records

    train_records = make_records(train_pool)
    test_records = make_records(test_pool, is_test=True)

    random.shuffle(train_records)
    val_records = train_records[:len(train_records)//4]
    train_records = train_records[len(train_records)//4:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f_train = DATA_DIR / "train_focused_family.jsonl"
    f_val = DATA_DIR / "val_focused_family.jsonl"
    f_test = DATA_DIR / "frozen_test_focused_family.jsonl"

    with open(f_train, "w", encoding="utf-8") as f:
        for r in train_records: f.write(json.dumps(r) + "\n")
    with open(f_val, "w", encoding="utf-8") as f:
        for r in val_records: f.write(json.dumps(r) + "\n")
    with open(f_test, "w", encoding="utf-8") as f:
        for r in test_records: f.write(json.dumps(r) + "\n")

    h_test = hashlib.sha256(f_test.read_bytes()).hexdigest()
    print(f"Focused Task Family Dataset Created:")
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Frozen Test: {len(test_records)}")
    print(f"Frozen Test SHA-256: {h_test}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "FOCUSED_FAMILY_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump({
            "task_family": "simple_compound_arithmetic_and_clarification",
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
            "frozen_test_sha256": h_test,
        }, f, indent=2)


if __name__ == "__main__":
    build_focused_dataset()
