"""Step 3: Diagnostic Evaluation Across Parent, Exp 01, and Exp 02 Checkpoints.

Evaluates a curated diagnostic suite spanning:
A. Exact training examples (seen during Exp 02).
B. Validation examples (seen status values in val split).
C. Frozen held-out test examples (unseen status values).
D. Short single-word vs longer multi-word status values.
E. Multiple prompt templates.

Measures:
- Autoregressive greedy generation (exact match, parseability, schema, output text).
- Teacher-forced next-token log probabilities and cross-entropy loss across the supervised response.
- First-token and top-k logit predictions at key decision points (e.g. at the start of the status string).
- Direct comparison between teacher-forced accuracy vs free autoregressive generation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset
from experiments.research_v08_json_sft.exp02_string_copy.evaluate_exp02 import (
    generate_greedy,
    evaluate_output_case,
)

OUTPUT_FILE = Path(__file__).resolve().parent / "diagnostic_evaluation_results.json"


def run_diagnostic_suite():
    print("=" * 70)
    print("STEP 3: MULTI-CHECKPOINT DIAGNOSTIC EVALUATION SUITE")
    print("=" * 70)

    device = torch.device("cpu")
    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    # Load 3 checkpoints
    parent_path = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    exp01_path = REPO_ROOT / "experiments" / "research_v08_json_sft" / "checkpoints" / "sft_best_checkpoint.pt"
    exp02_path = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt"

    models = {}
    for name, path in [("Canonical Parent", parent_path), ("Exp01 Candidate", exp01_path), ("Exp02 Candidate", exp02_path)]:
        m = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
        st = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
        m.eval()
        models[name] = m
        print(f"Loaded {name} from {path.name}")

    # Build diagnostic cases
    train_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl"
    val_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "val.jsonl"
    test_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "test.jsonl"

    with open(train_file, "r", encoding="utf-8") as f:
        train_pool = [json.loads(line) for line in f if "single_field" in line]
    with open(val_file, "r", encoding="utf-8") as f:
        val_pool = [json.loads(line) for line in f if "single_field" in line]
    with open(test_file, "r", encoding="utf-8") as f:
        test_pool = [json.loads(line) for line in f if "single_field" in line]

    diagnostic_items = [
        # A. Training examples (seen status values)
        {"subset": "Train", "desc": "Short status (nominal)", "record": next(r for r in train_pool if r["expected_status"] == "nominal")},
        {"subset": "Train", "desc": "Short status (active)", "record": next(r for r in train_pool if r["expected_status"] == "active")},
        {"subset": "Train", "desc": "Compound status (carrier_locked)", "record": next(r for r in train_pool if r["expected_status"] == "carrier_locked")},
        {"subset": "Train", "desc": "Compound status (bearing_greased)", "record": next(r for r in train_pool if r["expected_status"] == "bearing_greased")},
        
        # B. Validation examples (unseen in training, seen in validation)
        {"subset": "Val", "desc": "Compound status (standby_normal)", "record": next(r for r in val_pool if r["expected_status"] == "standby_normal")},
        {"subset": "Val", "desc": "Compound status (frequency_drift)", "record": next(r for r in val_pool if r["expected_status"] == "frequency_drift")},
        {"subset": "Val", "desc": "Compound status (voltage_spike)", "record": next(r for r in val_pool if r["expected_status"] == "voltage_spike")},
        {"subset": "Val", "desc": "Compound status (thermal_gradient)", "record": next(r for r in val_pool if r["expected_status"] == "thermal_gradient")},

        # C. Frozen held-out test examples (completely unseen)
        {"subset": "Test", "desc": "Long compound (optical_alignment_fault)", "record": next(r for r in test_pool if r["expected_status"] == "optical_alignment_fault")},
        {"subset": "Test", "desc": "Compound status (beam_misaligned)", "record": next(r for r in test_pool if r["expected_status"] == "beam_misaligned")},
        {"subset": "Test", "desc": "Compound status (cavity_depressurized)", "record": next(r for r in test_pool if r["expected_status"] == "cavity_depressurized")},
        {"subset": "Test", "desc": "Short compound (cryo_quench)", "record": next(r for r in test_pool if r["expected_status"] == "cryo_quench")},
    ]

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    all_results = []

    for item in diagnostic_items:
        rec = item["record"]
        subset = item["subset"]
        desc = item["desc"]
        prompt = rec["prompt"]
        expected_resp = rec["response"]
        expected_status = rec["expected_status"]

        prompt_str = prompt if prompt.endswith("\n") else prompt.strip()
        p_toks = tokenizer.encode(prompt_str)
        input_seq = [0] + p_toks

        # Build teacher-forced causal sequence for loss calculation
        # input_seq_tf: [BOS] + p_toks + r_toks
        # target_seq_tf: [-100]*len(p_toks) + r_toks + [EOS]
        r_toks = tokenizer.encode(expected_resp)
        tf_input = [0] + p_toks + r_toks
        tf_target = ([-100] * len(p_toks)) + r_toks + [1]
        assert len(tf_input) == len(tf_target)
        resp_start_pos = len(p_toks)

        item_eval = {
            "subset": subset,
            "description": desc,
            "expected_status": expected_status,
            "expected_response": expected_resp,
            "models": {}
        }

        print(f"\n[{subset}] {desc} | Status: '{expected_status}'")

        for m_name, model in models.items():
            # 1. Greedy Autoregressive Generation
            gen_out = generate_greedy(model, tokenizer, input_seq, max_new_tokens=40, context_limit=256)
            eval_case = evaluate_output_case(
                "single_field_string_extraction",
                gen_out["text"],
                expected_status=expected_status,
                is_empty=gen_out["is_empty"],
                first_token_is_eos=gen_out["first_token_is_eos"]
            )

            # 2. Teacher-Forced Loss & Next-Token Top-k at First Status Token
            with torch.no_grad():
                tf_tensor = torch.tensor([tf_input], dtype=torch.long)
                logits = model(tf_tensor)[0]  # [seq_len, vocab_size]

            # Compute cross entropy per token across response
            tf_losses = []
            tf_correct_tokens = 0
            # Target at pos `i` is predicted by `tf_input[i]`
            for pos in range(resp_start_pos, len(tf_input)):
                tgt_t = tf_target[pos]
                l = loss_fn(logits[pos:pos+1, :], torch.tensor([tgt_t])).item()
                pred_t = int(torch.argmax(logits[pos, :]).item())
                tf_losses.append(l)
                if pred_t == tgt_t:
                    tf_correct_tokens += 1

            mean_tf_loss = round(sum(tf_losses) / max(1, len(tf_losses)), 4)
            tf_acc = round(tf_correct_tokens / max(1, len(tf_losses)) * 100, 1)

            # Find position where the status value begins (after `{"status": "`)
            # `{"status": "` tokens in response:
            # Let's inspect logits at the token right before status value
            # In single_field_extraction, response begins: `{"status": "`
            prefix_toks = tokenizer.encode('{"status": "')
            status_start_pos = resp_start_pos + len(prefix_toks) - 1
            if status_start_pos < len(tf_input):
                status_logits = logits[status_start_pos, :]
                top5_vals, top5_inds = torch.topk(status_logits, 5)
                top5_list = [(tokenizer.decode([int(ind.item())]), round(val.item(), 2)) for ind, val in zip(top5_inds, top5_vals)]
            else:
                top5_list = []

            model_record = {
                "raw_output": gen_out["text"],
                "json_parsed": eval_case["json_parsed"],
                "schema_valid": eval_case["schema_valid"],
                "status_match": eval_case["status_match"],
                "mean_teacher_forced_loss": mean_tf_loss,
                "teacher_forced_token_accuracy_pct": tf_acc,
                "top5_first_status_token": top5_list,
                "stop_reason": gen_out["stop_reason"],
            }
            item_eval["models"][m_name] = model_record

            print(f"  {m_name:18s} | Parse: {str(eval_case['json_parsed']):5s} | Schema: {str(eval_case['schema_valid']):5s} | Match: {str(eval_case['status_match']):5s} | TF Loss: {mean_tf_loss:.2f} | TF Acc: {tf_acc:4.1f}% | Out: {repr(gen_out['text'])}")

        all_results.append(item_eval)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDiagnostic results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_diagnostic_suite()
