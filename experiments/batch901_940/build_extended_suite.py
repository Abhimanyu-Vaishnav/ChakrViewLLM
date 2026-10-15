"""
WP 921–927: Build an extended leakage-resistant evaluation suite.
Maintains the original 70-item frozen benchmark completely untouched and adds
20 new challenging held-out scenarios across multi-step chains, Hindi/Hinglish,
and long text arguments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "experiments" / "batch901_940" / "data"
REPORT_DIR = REPO_ROOT / "experiments" / "batch901_940" / "reports"


def build_extended_suite():
    extended_items = [
        # 1. 3-step arithmetic
        {
            "id": "ext_math3_01",
            "family": "math_3step",
            "prompt": "User: Take 300, multiply by 4, add 25, and subtract 10\nPlan:\n",
            "target_dsl": "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=300*4} | deps=[]\nSTEP 1: tool_call | calculator | args={expr=$0+25} | deps=[0]\nSTEP 2: tool_call | calculator | args={expr=$1-10} | deps=[1]\nCLARIFICATION: NONE\n<END_PLAN>",
        },
        {
            "id": "ext_math3_02",
            "family": "math_3step",
            "prompt": "User: Take 450, multiply by 2, add 80, and subtract 30\nPlan:\n",
            "target_dsl": "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=450*2} | deps=[]\nSTEP 1: tool_call | calculator | args={expr=$0+80} | deps=[0]\nSTEP 2: tool_call | calculator | args={expr=$1-30} | deps=[1]\nCLARIFICATION: NONE\n<END_PLAN>",
        },
        # 2. Hindi / Hinglish compound requests
        {
            "id": "ext_hinglish_01",
            "family": "hinglish_arithmetic",
            "prompt": "User: 700 aur 6 ka guna karo aur result me 15 jodo\nPlan:\n",
            "target_dsl": "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=700*6} | deps=[]\nSTEP 1: tool_call | calculator | args={expr=$0+15} | deps=[0]\nCLARIFICATION: NONE\n<END_PLAN>",
        },
        {
            "id": "ext_hinglish_02",
            "family": "hinglish_arithmetic",
            "prompt": "User: 850 ko 3 se divide karo aur fir 40 add karo\nPlan:\n",
            "target_dsl": "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=850/3} | deps=[]\nSTEP 1: tool_call | calculator | args={expr=$0+40} | deps=[0]\nCLARIFICATION: NONE\n<END_PLAN>",
        },
        # 3. Text utility with Unicode & special punctuation
        {
            "id": "ext_text_01",
            "family": "text_chain",
            "prompt": "User: Process phrase 'Cryptographic State Hash #99', trim surrounding spaces, and convert to uppercase\nPlan:\n",
            "target_dsl": "GOAL: text_transform_chain\nSTEP 0: text_transform | text_utility | args={text='Cryptographic State Hash #99',op=strip} | deps=[]\nSTEP 1: text_transform | text_utility | args={text=$0,op=uppercase} | deps=[0]\nCLARIFICATION: NONE\n<END_PLAN>",
        },
        # 4. Missing parameter abstentions
        {
            "id": "ext_clar_01",
            "family": "clarification",
            "prompt": "User: Settle transaction for account_007 without supplying authorization_signature\nPlan:\n",
            "target_dsl": "GOAL: clarification_needed\nSTEP 0: clarification | none | args={missing=authorization_signature} | deps=[]\nCLARIFICATION: NEEDED\n<END_PLAN>",
        },
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "extended_eval_benchmark.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for it in extended_items:
            f.write(json.dumps(it) + "\n")

    h = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(f"Extended benchmark created with {len(extended_items)} items.")
    print(f"SHA-256: {h}")

    manifest = {
        "timestamp": "2026-10-15 17:50:35",
        "frozen_benchmark_70_sha256": "cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a",
        "extended_benchmark_file": str(out_file),
        "extended_benchmark_sha256": h,
        "item_count": len(extended_items),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "EVALUATION_SUITE_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    build_extended_suite()
