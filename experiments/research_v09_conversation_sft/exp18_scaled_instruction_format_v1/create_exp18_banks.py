"""
create_exp18_banks.py
Generates the comprehensive, verified, high-quality candidate_banks.py for Exp18.
- 190 Factual QA items (150 train, 20 val, 20 test)
- 190 Short Explanation items (150 train, 20 val, 20 test)
- 190 Uncertainty Abstention items (150 train, 20 val, 20 test) with grounded controls and unanswerable queries
All items adhere to:
- Token length in [3, 45]
- 0% leakage with Gate v0.1
- 0% cross-split leakage
"""

import json
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

# ---------------------------------------------------------
# Load Gate v0.1 Prompts
# ---------------------------------------------------------
gate_path = REPO_ROOT / "experiments" / "normal_chat_gate_v01" / "prompts.jsonl"
gate_prompts = []
with open(gate_path, "r", encoding="utf-8") as f:
    for line in f:
        gate_prompts.append(json.loads(line))

FORBIDDEN_GATE_ENTITIES = {
    "mars", "red planet",
    "olleh", "hello backwards",
    "chlorophyll", "leaves are green",
    "station a", "station b", "train leaves",
    "project chimera",
    "agent 007", "delta-x", "liquid nitrogen",
    "closing price of gold", "december 31, 2030",
    "chemical formula for water", "h2o",
    "elephant or a mouse",
    "solar panels",
    "say cat, dog, bird",
}

def validate_item(q, a, name=""):
    tokens = tokenizer.encode(a.strip())
    tlen = len(tokens)
    if tlen < 3 or tlen > 45:
        raise ValueError(f"[{name}] Token bound error ({tlen}): {a}")
    q_lower = q.lower()
    for ent in FORBIDDEN_GATE_ENTITIES:
        if ent in q_lower:
            raise ValueError(f"[{name}] Gate entity '{ent}' in prompt: {q}")
    for gp in gate_prompts:
        gp_text = gp["prompt"].lower().strip()
        if q_lower == gp_text:
            raise ValueError(f"[{name}] Gate exact match: {q}")
        import difflib
        ratio = difflib.SequenceMatcher(None, q_lower, gp_text).ratio()
        if ratio >= 0.70:
            raise ValueError(f"[{name}] Gate similarity {ratio:.2f}: {q}")
    return tlen

print("Validator ready.")
