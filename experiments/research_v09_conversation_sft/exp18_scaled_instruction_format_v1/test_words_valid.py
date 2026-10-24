import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

candidate_sets = {
    "train": [
        "serendipity", "lighthouse", "quadrilateral", "microscope", "symphony",
        "caterpillar", "avalanche", "constellation", "kaleidoscope", "waterfall",
        "atmosphere", "archipelago", "thermometer", "pendulum", "hemisphere",
        "navigation", "manuscript", "peninsula"
    ],
    "val": [
        "stratosphere", "heliosphere", "gravitation", "astrophysics", "crystallography"
    ],
    "test": [
        "spectroscopy", "thermodynamics", "electromagnetism", "supernova", "interstellar"
    ]
}

for split, words in candidate_sets.items():
    for w in words:
        for resp in [w.upper(), w.lower(), w[::-1]]:
            tlen = len(tokenizer.encode(resp))
            if tlen < 3 or tlen > 45:
                print(f"FAIL: {split} '{resp}' -> {tlen} tokens")
print("Validation complete.")
