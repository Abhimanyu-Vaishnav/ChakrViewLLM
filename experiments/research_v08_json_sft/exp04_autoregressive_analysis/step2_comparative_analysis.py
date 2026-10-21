"""Step 2: Deep Comparative Analysis of Teacher-Forced vs Free-Running Generation.

Measures for each example in a balanced diagnostic subset (Train, Val, Held-Out Test):
1. Ground-truth response.
2. Overall teacher-forced token accuracy and per-token loss.
3. First status-value token analysis:
   - Target token ID, text, rank in model logits, predicted probability, and top-5 alternative tokens.
4. Free-running greedy output.
5. Earliest generation step where output diverges from ground truth.
6. Target subwords teacher-forced accuracy (specifically on the status string tokens).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp02_string_copy.evaluate_exp02 import (
    generate_greedy,
    evaluate_output_case,
)

OUTPUT_FILE = Path(__file__).resolve().parent / "step2_comparative_analysis.json"


def run_comparative_analysis():
    print("=" * 70)
    print("STEP 2: TEACHER-FORCED VS FREE-RUNNING COMPARATIVE ANALYSIS")
    print("=" * 70)

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    # Load Exp02 candidate checkpoint
    exp02_path = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt"
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    st = torch.load(exp02_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    # Load datasets
    train_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl"
    val_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "val.jsonl"
    test_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "test.jsonl"

    with open(train_file, "r", encoding="utf-8") as f:
        train_pool = [json.loads(line) for line in f if "single_field" in line]
    with open(val_file, "r", encoding="utf-8") as f:
        val_pool = [json.loads(line) for line in f if "single_field" in line]
    with open(test_file, "r", encoding="utf-8") as f:
        test_pool = [json.loads(line) for line in f if "single_field" in line]

    diagnostic_set = [
        # Train examples (seen status values)
        {"subset": "Train", "record": next(r for r in train_pool if r["expected_status"] == "nominal")},
        {"subset": "Train", "record": next(r for r in train_pool if r["expected_status"] == "active")},
        {"subset": "Train", "record": next(r for r in train_pool if r["expected_status"] == "carrier_locked")},
        {"subset": "Train", "record": next(r for r in train_pool if r["expected_status"] == "bearing_greased")},
        {"subset": "Train", "record": next(r for r in train_pool if r["expected_status"] == "discharged")},

        # Val examples (unseen in training, seen in validation)
        {"subset": "Val", "record": next(r for r in val_pool if r["expected_status"] == "standby_normal")},
        {"subset": "Val", "record": next(r for r in val_pool if r["expected_status"] == "frequency_drift")},
        {"subset": "Val", "record": next(r for r in val_pool if r["expected_status"] == "voltage_spike")},
        {"subset": "Val", "record": next(r for r in val_pool if r["expected_status"] == "thermal_gradient")},

        # Held-out test examples (completely unseen)
        {"subset": "Test", "record": next(r for r in test_pool if r["expected_status"] == "optical_alignment_fault")},
        {"subset": "Test", "record": next(r for r in test_pool if r["expected_status"] == "beam_misaligned")},
        {"subset": "Test", "record": next(r for r in test_pool if r["expected_status"] == "cavity_depressurized")},
        {"subset": "Test", "record": next(r for r in test_pool if r["expected_status"] == "flux_fluctuation")},
        {"subset": "Test", "record": next(r for r in test_pool if r["expected_status"] == "cryo_quench")},
    ]

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    results = []

    for item in diagnostic_set:
        rec = item["record"]
        subset = item["subset"]
        prompt = rec["prompt"]
        response = rec["response"]
        expected_status = rec["expected_status"]

        prompt_str = prompt if prompt.endswith("\n") else prompt.strip()
        p_toks = tok.encode(prompt_str)
        r_toks = tok.encode(response)

        # Free generation
        gen_out = generate_greedy(model, tok, [0] + p_toks, max_new_tokens=40, context_limit=256)
        eval_case = evaluate_output_case("single_field_string_extraction", gen_out["text"], expected_status=expected_status)

        # Teacher-forced pass
        tf_input = [0] + p_toks + r_toks
        tf_target = ([-100] * len(p_toks)) + r_toks + [1]
        assert len(tf_input) == len(tf_target)

        with torch.no_grad():
            logits = model(torch.tensor([tf_input], dtype=torch.long))[0]

        resp_start_pos = len(p_toks)  # input at len(p_toks) is last prompt token
        tf_losses = []
        tf_matches = []

        for pos in range(resp_start_pos, len(tf_input)):
            tgt = tf_target[pos]
            pred = int(torch.argmax(logits[pos, :]).item())
            l = loss_fn(logits[pos:pos+1, :], torch.tensor([tgt])).item()
            tf_losses.append(l)
            tf_matches.append(pred == tgt)

        total_tf_acc = round(sum(tf_matches) / len(tf_matches) * 100, 1)
        mean_tf_loss = round(sum(tf_losses) / len(tf_losses), 4)

        # Response format: '{"status": "' (5 tokens) + <status_tokens> + '"}' (2 tokens)
        # In tf_input, response tokens start at resp_start_pos:
        # pos resp_start_pos + 0: input is ':' -> target is '{'
        # pos resp_start_pos + 1: input is '{' -> target is '"'
        # pos resp_start_pos + 2: input is '"' -> target is 'stat'
        # pos resp_start_pos + 3: input is 'stat' -> target is 'us'
        # pos resp_start_pos + 4: input is 'us' -> target is '": "'
        # pos resp_start_pos + 5: input is '": "' -> target is FIRST STATUS SUBWORD!
        first_status_pos = resp_start_pos + 5
        first_status_target = tf_target[first_status_pos]
        first_status_logits = logits[first_status_pos, :]
        probs = F.softmax(first_status_logits, dim=-1)

        target_prob = round(probs[first_status_target].item(), 4)
        target_logit = round(first_status_logits[first_status_target].item(), 2)

        # Rank of target token (0-indexed)
        sorted_indices = torch.argsort(first_status_logits, descending=True)
        rank = int((sorted_indices == first_status_target).nonzero(as_tuple=True)[0].item()) + 1

        top5_inds = sorted_indices[:5]
        top5_list = [
            {
                "token_id": int(idx.item()),
                "token_str": tok.decode([int(idx.item())]),
                "logit": round(first_status_logits[idx].item(), 2),
                "prob": round(probs[idx].item(), 4),
            }
            for idx in top5_inds
        ]

        # Target subwords accuracy
        # Subwords span from first_status_pos to len(tf_input) - 3 (inclusive)
        # Because last 2 response tokens are '"', '}' followed by EOS target
        # tf_target has: [BOS] + p_toks + [ '{', '"', 'stat', 'us', '": "', <status_tokens>, '"', '}' ] + EOS
        status_tok_len = len(r_toks) - 7
        status_matches = tf_matches[4 : 4 + status_tok_len]
        status_losses = tf_losses[4 : 4 + status_tok_len]
        status_acc = round(sum(status_matches) / max(1, len(status_matches)) * 100, 1)
        status_loss = round(sum(status_losses) / max(1, len(status_losses)), 4)

        # Free generation divergence point
        gen_tokens = gen_out["generated_tokens"]
        expected_resp_tokens = r_toks + [1]
        divergence_step = None
        for step_idx in range(max(len(gen_tokens), len(expected_resp_tokens))):
            g_t = gen_tokens[step_idx] if step_idx < len(gen_tokens) else None
            e_t = expected_resp_tokens[step_idx] if step_idx < len(expected_resp_tokens) else None
            if g_t != e_t:
                divergence_step = step_idx
                break

        res_item = {
            "subset": subset,
            "status_value": expected_status,
            "expected_response": response,
            "free_output": gen_out["text"],
            "exact_match": eval_case["status_match"],
            "divergence_step": divergence_step,
            "total_tf_accuracy_pct": total_tf_acc,
            "mean_tf_loss": mean_tf_loss,
            "status_subwords_tf_accuracy_pct": status_acc,
            "status_subwords_tf_loss": status_loss,
            "first_status_token": {
                "target_id": first_status_target,
                "target_str": tok.decode([first_status_target]),
                "rank": rank,
                "target_prob": target_prob,
                "target_logit": target_logit,
                "top5_alternatives": top5_list,
            }
        }
        results.append(res_item)

        print(f"[{subset:5s}] Status: '{expected_status:24s}' | Gen: {repr(gen_out['text']):28s} | Diverge step: {divergence_step} | 1st Tok Rank: {rank:2d} (P={target_prob:.3f}) | Status TF Acc: {status_acc:4.1f}%")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nStep 2 results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_comparative_analysis()
