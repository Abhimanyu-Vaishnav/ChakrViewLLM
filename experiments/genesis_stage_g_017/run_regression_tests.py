"""
NR-017 Phase 8: Coding and Language Regression Suite
Evaluates:
1. Python coding execution correctness on 20 unseen programming tasks (AST parse + sandbox execution with assertions)
2. English, Hindi (Devanagari), and Hinglish dialogue regression check
3. Compares Condition A (Direct), Condition B (Explicit Trace), and Condition C (Compact Trace)
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

def run_coding_and_language_regression():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-017 CODING & LANGUAGE REGRESSION on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    # 1. Coding Tasks (20 tasks from coding_benchmark_50)
    with open("data/coding_eval_016/coding_benchmark_50.json", 'r', encoding='utf-8') as f:
        all_coding = json.load(f)
    coding_20 = all_coding[:20]

    # 2. Language Prompts (English, Hindi, Hinglish)
    lang_prompts = [
        {"id": "EN-01", "lang": "English", "prompt": "Hello! How can you help me today?"},
        {"id": "EN-02", "lang": "English", "prompt": "What is the difference between a list and a tuple?"},
        {"id": "HI-01", "lang": "Hindi", "prompt": "नमस्ते! आपका क्या नाम है?"},
        {"id": "HI-02", "lang": "Hindi", "prompt": "कंप्यूटर क्या होता है? संक्षेप में बताएं।"},
        {"id": "HG-01", "lang": "Hinglish", "prompt": "Bhai ek simple Python code likh kar dikhao."},
        {"id": "HG-02", "lang": "Hinglish", "prompt": "Data science sikhne ke liye kya zaroori hai?"}
    ]

    models = {
        "Direct_A": "Condition_A_Seed42.pt",
        "Explicit_B": "Condition_B_Seed42.pt",
        "Compact_C": "Condition_C_Seed42.pt"
    }

    regression_results = {}

    for m_name, ckpt_name in models.items():
        st = torch.load(f"experiments/genesis_stage_g_017/{ckpt_name}", map_location="cpu", weights_only=False)
        cfg = Genesis17MConfig(**st['config'])
        model = ChakrGenesis17M(cfg)
        model.load_state_dict(st['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"\n--- Testing Model: {m_name} ---")

        # Evaluate Coding
        syntax_valid = 0
        exec_passed = 0
        for task in coding_20:
            prompt = f"<user>\n{task['prompt']}\n<assistant>\n"
            p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            c = p_ids
            with torch.no_grad():
                for _ in range(80):
                    out, _ = model(c)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    c = torch.cat([c, nxt], dim=1)
                    if nxt.item() == 1: break
            code_out = tokenizer.decode(c[0, p_ids.size(1):].tolist()).strip()

            is_syntax = False
            is_pass = False
            try:
                ast.parse(code_out)
                is_syntax = True
                syntax_valid += 1
            except Exception:
                pass

            if is_syntax:
                test_script = f"{code_out}\n\n{task['test_assertions']}\n"
                try:
                    res = subprocess.run([sys.executable, "-c", test_script], capture_output=True, timeout=2.0)
                    if res.returncode == 0:
                        exec_passed += 1
                        is_pass = True
                except Exception:
                    pass

        print(f"  Coding (20 tasks): Syntax Valid = {syntax_valid}/20 ({syntax_valid/20*100:.1f}%) | Exec Correct = {exec_passed}/20 ({exec_passed/20*100:.1f}%)")

        # Evaluate Language
        utf8_clean = 0
        coherent = 0
        lang_outputs = []
        for lp in lang_prompts:
            prompt = f"<user>\n{lp['prompt']}\n<assistant>\n"
            p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            c = p_ids
            with torch.no_grad():
                for _ in range(40):
                    out, _ = model(c)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    c = torch.cat([c, nxt], dim=1)
                    if nxt.item() == 1: break
            resp = tokenizer.decode(c[0, p_ids.size(1):].tolist()).strip()

            if "\ufffd" not in resp: utf8_clean += 1
            words = resp.split()
            rep_ratio = len(set(words)) / max(len(words), 1)
            is_cohere = len(resp) > 3 and rep_ratio > 0.3
            if is_cohere: coherent += 1

            lang_outputs.append({
                "id": lp['id'],
                "lang": lp['lang'],
                "prompt": lp['prompt'],
                "response": resp,
                "coherent": is_cohere
            })

        print(f"  Language (6 prompts): UTF-8 Clean = {utf8_clean}/6 | Coherent = {coherent}/6")

        regression_results[m_name] = {
            "coding_syntax_valid": syntax_valid,
            "coding_exec_passed": exec_passed,
            "utf8_clean_rate": (utf8_clean / len(lang_prompts)) * 100,
            "coherence_rate": (coherent / len(lang_prompts)) * 100,
            "lang_samples": lang_outputs
        }

    out_file = Path("data/stage_g_017/regression_evaluation_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(regression_results, f, indent=2, ensure_ascii=False)

    print(f"\nRegression evaluation results saved to: {out_file}")

if __name__ == '__main__':
    run_coding_and_language_regression()
