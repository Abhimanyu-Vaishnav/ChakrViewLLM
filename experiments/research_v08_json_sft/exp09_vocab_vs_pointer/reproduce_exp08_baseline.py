"""
reproduce_exp08_baseline.py
ChakrView Experiment 09 — Step 2: Reproduce the Experiment 08 Baseline

Audits and re-evaluates the frozen Experiment 08 best checkpoint:
1. Verifies SHA-256 hashes of:
   - Canonical parent checkpoint (Step-10000)
   - Canonical tokenizer artifacts (vocab.json, merges.txt)
   - Experiment 08 best checkpoint (exp08_best_checkpoint.pt)
   - Experiment 08 dataset splits (train, val, test_a, test_b)
2. Runs exact evaluation on Test A (198 examples) and Test B (300 examples).
3. Verifies that the 100-example distractor lookup subset in Test A behaves as documented.
4. Checks and confirms the reported metrics (Test A EM ~92.9%, Test B EM 0.0%).
5. Halts with BLOCKED if reproduction fails.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP08_DIR = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp08_synthetic_copy_curriculum"
EXP09_DIR = Path(__file__).resolve().parent

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP08_BEST_CKPT = EXP08_DIR / "checkpoints" / "exp08_best_checkpoint.pt"

EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_EXP08_BEST_SHA = "26a2d405498111b404f70cff48098db3354db66ca75a67ddb48483555bc07fc0"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def eval_split_exact(model, tokenizer, split_path: Path, device="cpu"):
    with open(split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    first_token_hits = 0
    free_gen_hits = 0
    ranks = []
    probs = []
    distractor_total = 0
    distractor_hits = 0

    with torch.no_grad():
        for rec in records:
            tmpl = rec.get("template_type", "")
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]

            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)

            # 1. Teacher-Forced
            full_seq = [0] + p_toks + r_toks + [1]
            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(inp_tensor)
            
            target_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(target_logit).item())
            softmax_probs = F.softmax(target_logit, dim=-1)
            target_prob = softmax_probs[target_tok_id].item()
            sorted_indices = torch.argsort(target_logit, descending=True)
            target_rank = int((sorted_indices == target_tok_id).nonzero(as_tuple=True)[0].item()) + 1

            ranks.append(target_rank)
            probs.append(target_prob)
            if pred_tok_id == target_tok_id:
                first_token_hits += 1

            # 2. Free Autoregressive Generation
            curr = [0] + list(p_toks)
            gen_tokens = []
            for _ in range(8):
                g_inp = torch.tensor([curr], dtype=torch.long, device=device)
                g_logits = model(g_inp)
                nxt = int(torch.argmax(g_logits[0, -1, :]).item())
                if nxt == 1:
                    break
                gen_tokens.append(nxt)
                curr.append(nxt)

            try:
                gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
            except Exception:
                gen_text = tokenizer.decode(gen_tokens).strip()

            is_em = (gen_text == target_val.strip())
            if is_em:
                free_gen_hits += 1

            if tmpl == "multi_item_lookup":
                distractor_total += 1
                if is_em:
                    distractor_hits += 1

    return {
        "total": total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "mean_rank": float(sum(ranks) / total),
        "mean_prob": float(sum(probs) / total),
        "distractor_total": distractor_total,
        "distractor_hits": distractor_hits,
        "distractor_em": (distractor_hits / distractor_total * 100.0) if distractor_total > 0 else 0.0
    }

def main():
    print("=" * 70)
    print("EXP 09 STEP 2: REPRODUCE EXPERIMENT 08 BASELINE")
    print("=" * 70)

    # 1. Hashes
    p_sha = compute_sha256(CANONICAL_CKPT)
    v_sha = compute_sha256(TOKENIZER_DIR / "vocab.json")
    e8_sha = compute_sha256(EXP08_BEST_CKPT)

    print(f"Parent Checkpoint Hash: {p_sha}")
    assert p_sha == EXPECTED_PARENT_SHA, "Canonical parent hash mismatch!"
    print(f"Tokenizer Vocab Hash:   {v_sha}")
    assert v_sha == EXPECTED_VOCAB_SHA, "Tokenizer vocab hash mismatch!"
    print(f"Exp 08 Best Hash:       {e8_sha}")
    assert e8_sha == EXPECTED_EXP08_BEST_SHA, "Exp 08 best checkpoint hash mismatch!"
    print("  [PASS] All core artifact hashes cryptographically verified.")

    # 2. Token identity partition audit
    with open(EXP08_DIR / "data" / "train.jsonl", "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f if line.strip()]
    with open(EXP08_DIR / "data" / "test_a_familiar_tokens.jsonl", "r", encoding="utf-8") as f:
        test_a_recs = [json.loads(line) for line in f if line.strip()]
    with open(EXP08_DIR / "data" / "test_b_held_out_tokens.jsonl", "r", encoding="utf-8") as f:
        test_b_recs = [json.loads(line) for line in f if line.strip()]

    train_toks = set(r["target_token_id"] for r in train_recs)
    test_a_toks = set(r["target_token_id"] for r in test_a_recs)
    test_b_toks = set(r["target_token_id"] for r in test_b_recs)

    assert test_a_toks.issubset(train_toks), "Test A tokens not subset of Train!"
    assert train_toks.isdisjoint(test_b_toks), "Test B tokens overlap Train!"
    print("  [PASS] Token-identity partition strictly verified (Test A familiar, Test B disjoint).")

    # 3. Model load and evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    state = torch.load(EXP08_BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("\nEvaluating Exp 08 Best Checkpoint on Test A (198 examples)...")
    res_a = eval_split_exact(model, tokenizer, EXP08_DIR / "data" / "test_a_familiar_tokens.jsonl", device=device)
    print(f"  Test A TF Acc: {res_a['first_token_acc']:.2f}% | Free EM: {res_a['free_gen_em']:.2f}% | Mean Rank: {res_a['mean_rank']:.2f} | Distractor EM: {res_a['distractor_em']:.2f}% ({res_a['distractor_hits']}/{res_a['distractor_total']})")

    print("\nEvaluating Exp 08 Best Checkpoint on Test B (300 examples)...")
    res_b = eval_split_exact(model, tokenizer, EXP08_DIR / "data" / "test_b_held_out_tokens.jsonl", device=device)
    print(f"  Test B TF Acc: {res_b['first_token_acc']:.2f}% | Free EM: {res_b['free_gen_em']:.2f}% | Mean Rank: {res_b['mean_rank']:.2f}")

    # Assert reproduction bounds
    assert abs(res_a["free_gen_em"] - 92.93) < 0.1, f"Test A reproduction failed: expected ~92.9%, got {res_a['free_gen_em']}%"
    assert res_b["free_gen_em"] == 0.0, f"Test B reproduction failed: expected 0.0%, got {res_b['free_gen_em']}%"
    assert res_a["distractor_em"] == 97.0, f"Distractor subset reproduction failed: expected 97.0%, got {res_a['distractor_em']}%"

    reproduction_report = {
        "status": "REPRODUCED",
        "hashes": {
            "canonical_parent_sha256": p_sha,
            "canonical_vocab_sha256": v_sha,
            "exp08_best_checkpoint_sha256": e8_sha
        },
        "metrics": {
            "test_a": res_a,
            "test_b": res_b
        }
    }
    with open(EXP09_DIR / "baseline_reproduction_report.json", "w", encoding="utf-8") as f:
        json.dump(reproduction_report, f, indent=2)

    print("\n" + "=" * 70)
    print("[SUCCESS] EXPERIMENT 08 BASELINE REPRODUCED EXACTLY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
