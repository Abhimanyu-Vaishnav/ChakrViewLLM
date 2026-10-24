import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

test_words = [
    "nebula", "pulsar", "quasar", "supernova", "interstellar", "stratosphere",
    "heliosphere", "magnetosphere", "gravitation", "astrophysics", "crystallography",
    "photosynthesis", "spectroscopy", "thermodynamics", "electromagnetism"
]

for w in test_words:
    for resp in [w.upper(), w.lower(), w[::-1]]:
        tlen = len(tokenizer.encode(resp))
        print(f"'{resp}' -> {tlen} tokens")
