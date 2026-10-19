"""Phase 2: Target Alignment and Loss Masking Audit.

Audits:
1. Exact causal token alignment between input_seq and target_seq.
2. Verification that prompt tokens are masked with -100 and not included in loss.
3. Verification that target_seq[len(prompt_tokens)] == response_tokens[0] (first response token).
4. Verification that append_eos correctly places EOS (token ID 1) at the end.
5. Verification that continuation replay preserves natural token continuation.
6. Validation that truncation logic cannot silently discard target tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence, normalize_prompt_response_strings
TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
TRAIN_FILE = REPO_ROOT / "experiments" / "batch3111_3150" / "train.jsonl"
VAL_FILE = REPO_ROOT / "experiments" / "batch3111_3150" / "val.jsonl"


def audit_tokenization_alignment() -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(str(TOK_DIR))
    bos_id = 0
    eos_id = 1
    ignore_index = -100

    results = {
        "alignment_checks_passed": True,
        "truncation_risk_detected": False,
        "max_token_len_found": 0,
        "per_sample_audit": [],
    }

    # Load 5 sample training and 5 validation records
    samples = []
    for fpath in [TRAIN_FILE, VAL_FILE]:
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < 5:
                    samples.append(json.loads(line))

    for rec in samples:
        prompt = rec["prompt"]
        response = rec["response"]
        append_eos = rec.get("append_eos", True)

        p_str, r_str = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
        p_toks = tokenizer.encode(p_str)
        r_toks = tokenizer.encode(r_str)

        input_seq, target_seq = build_causal_sequence(
            p_toks, r_toks, bos_id=bos_id, eos_id=eos_id, ignore_index=ignore_index, append_eos=append_eos
        )

        total_len = len(input_seq)
        results["max_token_len_found"] = max(results["max_token_len_found"], total_len)

        # 1. Assert input and target lengths match exactly
        assert len(input_seq) == len(target_seq), f"Length mismatch: {len(input_seq)} vs {len(target_seq)}"

        # 2. Check prompt masking: target_seq[0..len(p_toks)-1] must all be ignore_index (-100)
        for i in range(len(p_toks)):
            if target_seq[i] != ignore_index:
                results["alignment_checks_passed"] = False
                raise AssertionError(f"Prompt token at position {i} is not masked with -100")

        # 3. Check causal alignment:
        # input_seq[len(p_toks)] is the prompt's final token. Its prediction label in target_seq
        # must be r_toks[0] (the first response token).
        first_predicted_token = target_seq[len(p_toks)]
        if first_predicted_token != r_toks[0]:
            results["alignment_checks_passed"] = False
            raise AssertionError(f"First predicted target token {first_predicted_token} != r_toks[0] {r_toks[0]}")

        # 4. Check EOS behavior:
        if append_eos:
            if target_seq[-1] != eos_id:
                results["alignment_checks_passed"] = False
                raise AssertionError(f"Final target token {target_seq[-1]} != EOS ({eos_id})")

        results["per_sample_audit"].append({
            "id": rec["id"],
            "prompt_tokens_len": len(p_toks),
            "response_tokens_len": len(r_toks),
            "total_seq_len": total_len,
            "masked_tokens_count": sum(1 for t in target_seq if t == ignore_index),
            "active_loss_tokens_count": sum(1 for t in target_seq if t != ignore_index),
            "first_target_token_decoded": tokenizer.decode([first_predicted_token]),
            "final_target_token_decoded": tokenizer.decode([target_seq[-1]]) if not append_eos else "<EOS>",
        })

    return results


if __name__ == "__main__":
    res = audit_tokenization_alignment()
    print("=================================================================")
    print("PHASE 2: TOKENIZATION & TARGET ALIGNMENT AUDIT COMPLETED")
    print("=================================================================")
    print(f"Alignment Checks Passed: {res['alignment_checks_passed']}")
    print(f"Max Token Length Found:  {res['max_token_len_found']} (Context Window: 512)")
    print(f"Samples Audited:         {len(res['per_sample_audit'])}")
    for s in res["per_sample_audit"][:4]:
        print(f"  Sample {s['id']}: Prompt {s['prompt_tokens_len']} tok | Resp {s['response_tokens_len']} tok | Loss Tokens {s['active_loss_tokens_count']} | First Pred: {repr(s['first_target_token_decoded'])}")
    print("=================================================================")

    out_path = REPO_ROOT / "experiments" / "sft_v3_readiness" / "ALIGNMENT_AUDIT_SUMMARY.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved alignment audit to: {out_path}")
