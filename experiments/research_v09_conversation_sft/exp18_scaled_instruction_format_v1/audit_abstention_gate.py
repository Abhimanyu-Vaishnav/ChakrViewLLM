import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.candidate_banks import CANDIDATE_ABSTENTION_BANKS
from experiments.research_v09_conversation_sft.exp17_instruction_sft_v1.generate_exp17_datasets import FORBIDDEN_GATE_ENTITIES

for split, items in CANDIDATE_ABSTENTION_BANKS.items():
    for idx, (ctx, uq, ua, cq, ca, dom) in enumerate(items):
        full_text = f"{ctx} {uq} {ua} {cq} {ca}".lower()
        for ent in FORBIDDEN_GATE_ENTITIES:
            if ent in full_text:
                print(f"{split} item {idx}: entity '{ent}' found in:\n  {ctx}")
