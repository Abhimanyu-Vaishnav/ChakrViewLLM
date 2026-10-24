import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import difflib
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.raw_exp_data import EXPLANATION_DATA
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_candidate_banks_exp18 import FORBIDDEN_GATE_ENTITIES, gate_prompts

for split, items in EXPLANATION_DATA.items():
    for q, a, intent, note in items:
        p_lower = q.lower().strip()
        for ent in FORBIDDEN_GATE_ENTITIES:
            if ent in p_lower:
                print(f"Gate entity violation [{split}]: {ent} in '{q}'")
        for gp in gate_prompts:
            ratio = difflib.SequenceMatcher(None, p_lower, gp['prompt'].lower().strip()).ratio()
            if ratio >= 0.70:
                print(f"Similarity violation [{split}]: {ratio:.2f} with {gp['id']} in '{q}'")
