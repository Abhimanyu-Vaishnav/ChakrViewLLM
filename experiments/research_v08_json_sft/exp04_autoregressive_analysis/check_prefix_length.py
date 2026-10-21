import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

prefix = '{"status": "'
prefix_toks = tok.encode(prefix)
print(f"Prefix {repr(prefix)} encodes to: {prefix_toks} ({[tok.decode([t]) for t in prefix_toks]})")
print(f"Length of prefix_toks: {len(prefix_toks)}")

# Sample response:
resp = '{"status": "discharged"}'
resp_toks = tok.encode(resp)
print(f"Response {repr(resp)} encodes to: {resp_toks} ({[tok.decode([t]) for t in resp_toks]})")
print(f"Prefix in response is the first {len(prefix_toks)} tokens: {resp_toks[:len(prefix_toks)] == prefix_toks}")
