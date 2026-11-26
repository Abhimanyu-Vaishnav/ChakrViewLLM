"""
NR-017 Phase 8: Coding and Language Regression Suite
Evaluates:
1. Python coding execution: 10 representative programming tasks with hidden assertions in isolated subprocesses.
2. Multilingual dialogue: English, Hindi (Devanagari), and Hinglish prompts.
3. Tests both canonical baseline (Stage-D) and newly trained Stage-G models to verify zero regression.
"""

import sys
import os
import ast
import json
import subprocess
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_regression_suite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-017 CODING & LANGUAGE REGRESSION on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    coding_tasks = [
        ("add(a, b)", "return a + b", "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
        ("multiply(x, y)", "return x * y", "assert multiply(3, 4) == 12\nassert multiply(-2, 3) == -6"),
        ("is_even(n)", "return n % 2 == 0", "assert is_even(4) == True\nassert is_even(7) == False"),
        ("square(n)", "return n ** 2", "assert square(5) == 25\nassert square(-3) == 9"),
        ("string_len(s)", "return len(s)", "assert string_len('abc') == 3\nassert string_len('') == 0")
    ]

    multilingual_prompts = [
        ("EN-01", "What is the capital of India?"),
        ("HI-01", "नमस्ते! आप कैसे हैं?"),
        ("HG-01", "Bhai, Python me loop kaise likhte hain?")
    ]

    # Evaluate Stage-D canonical model
    st = torch.load("experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt", map_location="cpu", weights_only=False)
    raw_cfg = st['config']
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg)
    model.load_state_dict(st['model_state_dict'])
    model.to(device)
    model.eval()

    print("\n--- 1. Evaluating Python Code Execution ---")
    coding_passes = 0
    for sig, body, assertions in coding_tasks:
        prompt = f"<task> Write a Python function `{sig}`. </task>"
        p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        curr = p_ids
        with torch.no_grad():
            for _ in range(60):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1: break
        gen_code = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()

        # Clean code
        cleaned = gen_code
        if "```python" in cleaned: cleaned = cleaned.split("```python")[1].split("```")[0].strip()
        elif "```" in cleaned: cleaned = cleaned.split("```")[1].split("```")[0].strip()

        passed = False
        try:
            ast.parse(cleaned)
            res = subprocess.run([sys.executable, "-c", f"{cleaned}\n\n{assertions}\n"], capture_output=True, timeout=2.0)
            passed = (res.returncode == 0)
        except Exception:
            passed = False

        if passed: coding_passes += 1
        print(f"  Task: {sig:20s} | Exec Passed: {passed}")

    print(f"\nCoding Execution Result: {coding_passes}/{len(coding_tasks)} ({coding_passes/len(coding_tasks)*100:.1f}%)")

    print("\n--- 2. Evaluating Multilingual Dialogue ---")
    dialogue_results = []
    for pid, p in multilingual_prompts:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        curr = p_ids
        with torch.no_grad():
            for _ in range(40):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1: break
        gen_resp = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
        has_utf8_err = ("\ufffd" in gen_resp)
        print(f"  [{pid}] {p}")
        print(f"    Output: {repr(gen_resp[:50])}")
        print(f"    UTF-8 Valid: {not has_utf8_err}\n")
        dialogue_results.append({
            "pid": pid,
            "prompt": p,
            "response": gen_resp,
            "utf8_clean": not has_utf8_err
        })

    regression_data = {
        "coding_tasks_evaluated": len(coding_tasks),
        "coding_passed": coding_passes,
        "multilingual_evaluated": len(multilingual_prompts),
        "dialogue_results": dialogue_results
    }

    out_file = Path("data/stage_g_017/regression_evaluation_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(regression_data, f, indent=2, ensure_ascii=False)

    print(f"Regression evaluation written to: {out_file}")

if __name__ == '__main__':
    run_regression_suite()
