import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.debug_5grams import all_overlaps

all_unique_5grams = sorted(list(set(" ".join(g) for g, p in all_overlaps)))
print(f"Total unique overlapping 5-grams: {len(all_unique_5grams)}")
for ug in all_unique_5grams:
    print(f'            "{ug}",')
