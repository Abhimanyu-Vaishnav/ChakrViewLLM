import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(Path('data/experiments/vocab_4096'))

with open('experiments/research_v08_json_sft/data/train.jsonl') as f:
    train_recs = [json.loads(line) for line in f if 'two_key' in line]

for r in train_recs[:4]:
    s = r['expected_status']
    prompt_str = r['prompt']
    target_str = r['response']
    
    s_toks = tok.encode(s)
    p_toks = tok.encode(prompt_str)
    t_toks = tok.encode(target_str)
    
    print(f"Status: {s}")
    print(f"  Isolated tokens: {s_toks} -> {[tok.decode([t]) for t in s_toks]}")
    # find where s_toks appears in p_toks and t_toks
    print(f"  Target tokens: {t_toks} -> {[tok.decode([t]) for t in t_toks]}")
