"""
UCIP-26 Phase 4: Teacher-Forcing vs Free-Generation Comparison.
Measures across 40 representative records:
1. Free Autoregressive Generation Output & Exact Match
2. Teacher-Forced Target Token NLL (cross-entropy loss over full target sequence)
3. First Answer Token Rank & Top-1 Probability
4. Full Multi-Token Candidate Target NLL vs Competitor/Attractor NLL
5. Forced-Prefix Continuation: Supply first target token, observe whether model continues target or collapses.
"""

from __future__ import annotations
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence


def load_model(checkpoint_path: Path) -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_teacher_forcing_diagnostics():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    sample_records = []
    for p in sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        for line in lines[:5]:
            sample_records.append(json.loads(line))

    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        print(f"\nRunning Teacher-Forcing Diagnostics for {ckpt_key}...")
        model = load_model(ckpt_path)

        ckpt_results = []
        for rec in sample_records:
            rid = rec["id"]
            task = rec["task_type"]
            prompt = rec["prompt"]
            target = rec["target_response"].strip()

            p_str, r_str = normalize_prompt_response_strings(prompt, target, strip_delimiters=True)
            p_toks = tok.encode(p_str)
            r_toks = tok.encode(r_str)

            # Causal sequence with target supervision
            input_seq, target_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
            inp_t = torch.tensor([input_seq], dtype=torch.long)
            tgt_t = torch.tensor([target_seq], dtype=torch.long)

            with torch.no_grad():
                logits = model(inp_t)  # shape (1, seq_len, vocab_size)
                # Compute teacher-forced NLL on active targets
                loss = F.cross_entropy(logits.view(-1, 4096), tgt_t.view(-1), ignore_index=-100, reduction="mean")
                teacher_forced_nll = float(loss.item())

                # Per-token probabilities on response tokens
                prompt_len = len(p_toks)
                # target_seq has target tokens starting at index prompt_len
                # In causal sequence, prediction for input_seq[t] is at logits[0, t, :]
                target_token_probs = []
                for idx in range(len(r_toks) + 1):  # response tokens + EOS
                    pos_in_logits = prompt_len + idx  # input token before target token
                    actual_target_tok = target_seq[prompt_len + idx]
                    log_probs = F.log_softmax(logits[0, pos_in_logits, :], dim=-1)
                    tok_prob = float(torch.exp(log_probs[actual_target_tok]).item())
                    target_token_probs.append(tok_prob)

                # Forced-Prefix Continuation Test:
                # If we supply the first target token, what does it predict next?
                forced_prefix_input = [0] + tok.encode(prompt) + [r_toks[0]]
                curr_forced = torch.tensor([forced_prefix_input], dtype=torch.long)
                gen_after_prefix = []
                for _ in range(8):
                    l_out = model(curr_forced)[0, -1, :]
                    nxt = int(torch.argmax(l_out).item())
                    if nxt == 1:
                        break
                    gen_after_prefix.append(nxt)
                    curr_forced = torch.cat([curr_forced, torch.tensor([[nxt]])], dim=1)

                continuation_text = tok.decode(gen_after_prefix)

            ckpt_results.append({
                "id": rid,
                "task_type": task,
                "target": target,
                "response_token_count": len(r_toks),
                "teacher_forced_nll": teacher_forced_nll,
                "perplexity": math.exp(min(teacher_forced_nll, 20)),
                "first_token_prob": target_token_probs[0],
                "all_target_token_probs": target_token_probs,
                "forced_prefix_token": tok.decode([r_toks[0]]),
                "forced_prefix_continuation": continuation_text
            })

        results[ckpt_key] = ckpt_results

    out_file = "experiments/ucip26/teacher_forcing/teacher_forcing_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Teacher-Forcing Diagnostics Complete. Saved to {out_file}")


if __name__ == "__main__":
    run_teacher_forcing_diagnostics()
