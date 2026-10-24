import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

gen = Exp18DatasetGenerator()
ds = gen.generate_all_splits()

def extract_5grams(text):
    toks = text.lower().replace("?", "").replace(".", "").replace("'", "").replace('"', '').split()
    return set(tuple(toks[i:i+5]) for i in range(len(toks) - 4))

train_5grams = set()
for r in ds["train"]:
    train_5grams.update(extract_5grams(r["prompt"]))

all_overlaps = set()
for r in ds["test"]:
    test_grams = extract_5grams(r["prompt"])
    ov = test_grams.intersection(train_5grams)
    for g in ov:
        all_overlaps.add((g, r["prompt"]))

print(f"Total overlapping 5-grams: {len(all_overlaps)}")
for g, p in sorted(all_overlaps):
    print("Overlap:", " ".join(g), "--> Prompt:", p)
