"""
generate_candidate_banks_exp18.py
Builds the complete candidate_banks.py for Exp18 with:
- 190 items in CANDIDATE_FACTUAL_BANKS (150 train, 20 val, 20 test)
- 190 items in CANDIDATE_EXPLANATION_BANKS (150 train, 20 val, 20 test)
- 190 items in CANDIDATE_ABSTENTION_BANKS (150 train, 20 val, 20 test)
Verifies token bounds [3, 45], Gate v0.1 disjointness, and writes to candidate_banks.py.
"""

import sys
import json
import difflib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

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

def check_gate(prompt, name=""):
    p_lower = prompt.lower().strip()
    for ent in FORBIDDEN_GATE_ENTITIES:
        if ent in p_lower:
            raise ValueError(f"[{name}] Forbidden Gate entity '{ent}' in '{prompt}'")
    for gp in gate_prompts:
        gp_text = gp["prompt"].lower().strip()
        if p_lower == gp_text:
            raise ValueError(f"[{name}] Exact match with Gate prompt {gp['id']}: '{prompt}'")
        ratio = difflib.SequenceMatcher(None, p_lower, gp_text).ratio()
        if ratio >= 0.70:
            raise ValueError(f"[{name}] Levenshtein ratio {ratio:.2f} >= 0.70 with Gate prompt {gp['id']}: '{prompt}'")

def check_tokens(text, name=""):
    tokens = tokenizer.encode(text.strip())
    tlen = len(tokens)
    if tlen < 3 or tlen > 45:
        raise ValueError(f"[{name}] Token bound violated ({tlen} tokens, allowed 3-45): '{text}'")
    return tlen

print("Sanity checks script loaded.")
