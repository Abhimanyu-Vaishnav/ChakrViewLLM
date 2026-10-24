import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

gen = Exp18DatasetGenerator()
ds = gen.generate_all_splits()
tr_p = set(r['prompt'].lower().strip() for r in ds['train'])
va_p = set(r['prompt'].lower().strip() for r in ds['val'])
overlap = tr_p.intersection(va_p)
for p in overlap:
    tr_matches = [r for r in ds['train'] if r['prompt'].lower().strip() == p]
    va_matches = [r for r in ds['val'] if r['prompt'].lower().strip() == p]
    print(f"Overlap prompt: '{p}'")
    print(f"  Train: id={tr_matches[0]['id']} cat={tr_matches[0]['category']}")
    print(f"  Val:   id={va_matches[0]['id']} cat={va_matches[0]['category']}")
