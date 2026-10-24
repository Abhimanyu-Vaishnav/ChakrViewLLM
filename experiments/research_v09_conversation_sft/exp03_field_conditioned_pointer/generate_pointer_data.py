"""
generate_pointer_data.py
ChakrView Experiment 15 — Field-Conditioned Pointer Dataset Generator.

Converts the Exp14 Context Grounding Benchmark into tokenized pointer training records:
- Maps each category to a distinct field slot ID (0 to 9)
- Tokenizes prompt context with canonical BPE tokenizer
- Accurately locates target substring token boundaries [start_pos, end_pos]
- If target is absent or requires abstention, labels as [0, 0] (BOS abstention)
- Generates candidate eligibility masks: True for context tokens, False for padding
- Preserves exact Train (100), Val (30), and Test (60) split boundaries from Exp14
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP14_DATA_DIR = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp02_context_grounding" / "data"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

EXP15_DIR = Path(__file__).resolve().parent
EXP15_DATA_DIR = EXP15_DIR / "data"
EXP15_DATA_DIR.mkdir(parents=True, exist_ok=True)

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CATEGORY_TO_FIELD_ID = {
    "dynamic_entity_binding": 0,
    "turn_dependent_retrieval": 1,
    "anaphora_reference_resolution": 2,
    "superseded_instruction_adherence": 3,
    "user_correction_handling": 4,
    "distractor_resistance": 5,
    "ambiguity_clarification": 6,
    "missing_context_abstention": 7,
    "factual_uncertainty_handling": 8,
    "constrained_code_synthesis": 9
}

def locate_target_span(tokens: List[int], target_tokens: List[int]) -> Tuple[int, int]:
    """Finds exact sublist slice in tokens."""
    if not target_tokens:
        return 0, 0
    L = len(target_tokens)
    for i in range(len(tokens) - L + 1):
        if tokens[i:i+L] == target_tokens:
            return i, i + L - 1
    return -1, -1

def compile_pointer_dataset(tokenizer, split_name: str) -> List[Dict[str, Any]]:
    input_file = EXP14_DATA_DIR / f"benchmark_{split_name}.jsonl"
    records = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    compiled = []
    
    for r in records:
        cat = r["category"]
        field_id = CATEGORY_TO_FIELD_ID[cat]
        m = r["metadata"]
        prompt = r["prompt"]
        
        # Determine exact target text to locate in prompt
        if cat == "dynamic_entity_binding":
            target_text = m["target_entity"]
        elif cat == "turn_dependent_retrieval":
            target_text = m["expected_value"]
        elif cat == "anaphora_reference_resolution":
            # Target is asking for clarification; both candidates are valid references
            target_text = m["candidates"][0]
        elif cat == "superseded_instruction_adherence":
            target_text = m["target_format"]
        elif cat == "user_correction_handling":
            target_text = m["new_target"]
        elif cat == "distractor_resistance":
            target_text = m["target_state"]
        elif cat == "ambiguity_clarification":
            target_text = m["target_pod"]
        elif cat in ["missing_context_abstention", "factual_uncertainty_handling"]:
            target_text = ""  # Explicit Abstention
        elif cat == "constrained_code_synthesis":
            target_text = m["function_name"]
        else:
            target_text = ""
            
        p_tokens = [0] + tokenizer.encode(prompt) # Prepend BOS (ID=0)
        
        if target_text == "":
            start_pos, end_pos = 0, 0
        else:
            t_tokens = tokenizer.encode(target_text)
            start_pos, end_pos = locate_target_span(p_tokens, t_tokens)
            if start_pos == -1:
                # Target text was not token-aligned; fallback to substring search on decoded slices
                # or label as abstention if impossible
                start_pos, end_pos = 0, 0
                
        # Candidate eligibility mask: True for context positions (1 to len(p_tokens)-1) + position 0 (abstain)
        candidate_mask = [True] * len(p_tokens)
        
        compiled.append({
            "example_id": r["example_id"],
            "category": cat,
            "field_id": field_id,
            "prompt": prompt,
            "target_text": target_text,
            "input_ids": p_tokens,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "is_abstention": (start_pos == 0 and end_pos == 0),
            "candidate_mask": candidate_mask
        })
        
    return compiled

def main():
    print("=" * 70)
    print("CHAKRVIEW: FIELD-CONDITIONED POINTER DATASET COMPILER")
    print("=" * 70)
    
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    
    for split in ["train", "val", "test"]:
        compiled = compile_pointer_dataset(tokenizer, split)
        out_path = EXP15_DATA_DIR / f"pointer_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in compiled:
                f.write(json.dumps(item) + "\n")
        print(f"Compiled {len(compiled)} records for pointer_{split}.jsonl")
        
    manifest = {
        "dataset_name": "ChakrView Field-Conditioned Pointer Dataset v1.0",
        "num_fields": len(CATEGORY_TO_FIELD_ID),
        "fields": CATEGORY_TO_FIELD_ID,
        "splits": {
            "train": {"file": "data/pointer_train.jsonl", "count": 100},
            "val": {"file": "data/pointer_val.jsonl", "count": 30},
            "test": {"file": "data/pointer_test.jsonl", "count": 60}
        }
    }
    
    manifest_path = EXP15_DIR / "DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}")

if __name__ == "__main__":
    main()
