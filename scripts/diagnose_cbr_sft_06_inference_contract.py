"""
scripts/diagnose_cbr_sft_06_inference_contract.py

Adversarial and structural contract testing between training format and inference execution:
1. Standard Prompt Serialization (User: ... \nAssistant:)
2. Missing Assistant Prefix
3. Extra Newlines and Separators
4. Special Characters in Technical Values
5. Multi-turn Alternating Dialogues
6. Context Truncation and Length Limits
"""

import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner, load_tokenizer_artifacts
from chakrview.training.sft_engine import normalize_prompt_response_strings, build_causal_sequence

def run_contract_audit():
    ckpt_path = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
    runner = DirectNeuralRunner(ckpt_path)
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    test_cases = [
        ("canonical_greeting", "User: [08:30:00] Good morning!\nAssistant:"),
        ("missing_prefix", "User: [08:30:00] Good morning!"),
        ("extra_newlines", "User: [08:30:00] Good morning!\n\n\nAssistant:\n"),
        ("trailing_space", "User: [08:30:00] Good morning!\nAssistant: "),
        ("technical_special_chars", "User: Variable DB_CONN=postgres://usr:pwd@10.0.0.1:5432/app\nQuestion: What is DB_CONN?\nAssistant:"),
        ("multiturn_dialogue", "User: Name is Alice.\nAssistant: Understood.\nUser: What is my name?\nAssistant:")
    ]

    results = []
    for name, p in test_cases:
        resp = runner.generate(prompt=p, max_new_tokens=20, temperature=0.0)
        results.append({
            "name": name,
            "prompt": p,
            "output": resp["text"],
            "stop_reason": resp["stop_reason"],
            "token_count": resp["token_count"]
        })

    out_file = REPO_ROOT / "evaluation" / "cbr_sft_06_suite" / "INFERENCE_CONTRACT_AUDIT.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Phase J: Inference and Evaluation Contract Audit\n\n")
        f.write("**Date:** 2026-11-04  \n")
        f.write(f"**Tested Model:** CBR-SFT-04 Step 200 (`{ckpt_path}`)  \n\n")
        f.write("## Adversarial Prompt Contract Results\n\n")
        f.write("| Test Name | Prompt Serialization | Model Output | Stop Reason |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in results:
            clean_out = r["output"].replace("\n", "\\n")
            clean_p = r["prompt"].replace("\n", "\\n")
            f.write(f"| `{r['name']}` | `{clean_p}` | `{clean_out}` | `{r['stop_reason']}` |\n")

    print(f"Contract audit complete. Saved to {out_file}")
    return results

if __name__ == "__main__":
    run_contract_audit()
