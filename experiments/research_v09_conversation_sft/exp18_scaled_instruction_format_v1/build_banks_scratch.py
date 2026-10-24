"""
generate_banks_data.py
Script to generate the deterministic, verified candidate banks for Exp18:
- factual_qa: 150 train, 20 val, 20 test (190 total)
- short_explanation: 150 train, 20 val, 20 test (190 total)
- uncertainty_abstention: 150 train, 20 val, 20 test (190 total, paired unanswerable + grounded control)
"""
import sys
from pathlib import Path

# Verify imports and token bounds
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)
print("Tokenizer loaded successfully. Vocab size:", tokenizer.vocab_size)
