"""Teacher-forced diagnostic engine for analyzing token prediction accuracy across JSON syntactic categories."""

import json
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CKPT_PATH = REPO_ROOT / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def eval_teacher_forcing(model, tok, path):
    with open(path, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    json_recs = [r for r in recs if r["category"].startswith("structured_output")]

    total_tokens = 0
    correct_tokens = 0
    token_category_stats = {
        "opening_brace": {"total": 0, "correct": 0},
        "key_tokens": {"total": 0, "correct": 0},
        "colon_comma": {"total": 0, "correct": 0},
        "values": {"total": 0, "correct": 0},
        "closing_brace": {"total": 0, "correct": 0},
        "eos_token": {"total": 0, "correct": 0},
    }

    for r in json_recs:
        p = r["prompt"]
        resp = r["response"]
        p_ids = [0] + tok.encode(p)
        r_ids = tok.encode(resp) + [1]  # EOS_ID is 1

        seq = p_ids + r_ids
        inp = torch.tensor([seq[:-1]], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp)
        preds = torch.argmax(logits[0], dim=-1)

        resp_start = len(p_ids) - 1
        for offset, tgt_tok in enumerate(r_ids):
            pred_tok = preds[resp_start + offset].item()
            is_corr = (pred_tok == tgt_tok)
            total_tokens += 1
            if is_corr:
                correct_tokens += 1

            tok_str = tok.decode([tgt_tok])
            if offset == 0:
                cat = "opening_brace"
            elif offset == len(r_ids) - 1:
                cat = "eos_token"
            elif "}" in tok_str:
                cat = "closing_brace"
            elif any(c in tok_str for c in [":", ","]):
                cat = "colon_comma"
            elif any(k in tok_str for k in ["status", "code", "metric", "value", "feature", "enabled", "name", "count"]):
                cat = "key_tokens"
            else:
                cat = "values"

            token_category_stats[cat]["total"] += 1
            if is_corr:
                token_category_stats[cat]["correct"] += 1

    acc = (correct_tokens / max(1, total_tokens)) * 100
    print(f"File {Path(path).name}: Total Tokens: {total_tokens}, Teacher-Forced Accuracy: {acc:.2f}%")
    for cat, s in token_category_stats.items():
        c_acc = (s["correct"] / max(1, s["total"])) * 100 if s["total"] > 0 else 0.0
        print(f"  {cat:18s}: {s['correct']:4d} / {s['total']:4d} ({c_acc:.1f}%)")

def main():
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("=== TEACHER-FORCED TOKEN DIAGNOSTIC (BATCH 3111-3150 STEP-60 CANDIDATE) ===")
    print("\n[TRAINING SPLIT - SEEN EXAMPLES]:")
    eval_teacher_forcing(model, tok, REPO_ROOT / "experiments" / "batch3111_3150" / "train.jsonl")
    print("\n[VALIDATION SPLIT - HELD-OUT EXAMPLES]:")
    eval_teacher_forcing(model, tok, REPO_ROOT / "experiments" / "batch3111_3150" / "val.jsonl")

if __name__ == "__main__":
    main()
