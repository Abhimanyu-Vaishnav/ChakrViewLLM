import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
print("Rhodium:", tok.encode("Rhodium"))
print(" Rhodium:", tok.encode(" Rhodium"))
print("35:", repr(tok.decode([35])))

