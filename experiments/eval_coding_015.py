"""
NR-015 Phase 7: Neural Coding Generalization Benchmark
Evaluates ChakrGenesis-17M Stage-D on 40 Python coding tasks with:
1. Syntax validity check (ast.parse)
2. Subprocess execution against hidden unit test assertions
3. Error type categorization (SyntaxError, AssertionError, NameError, Timeout)
"""

import sys
import os
import ast
import json
import subprocess
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M

def run_coding_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Coding Evaluation on: {device}")

    engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )
    model = engine.model
    tokenizer = engine.tokenizer
    model.eval()

    with open("data/sealed_eval_015/sealed_benchmark_300.json", 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)

    coding_tasks = [t for t in all_tasks if t['family'] == 'python_function_generation']
    print(f"Total coding tasks to evaluate: {len(coding_tasks)}")

    syntax_valid_count = 0
    assertion_passed_count = 0
    results = []

    for idx, task in enumerate(coding_tasks):
        prompt = f"<user>\n{task['prompt']}\n<assistant>\n"
        p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        curr = p_ids
        with torch.no_grad():
            for _ in range(120): # max 120 tokens
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1:
                    break

        gen_tokens = curr[0, p_ids.size(1):].tolist()
        gen_code = tokenizer.decode(gen_tokens).strip()

        # Clean markdown fences if present
        cleaned_code = gen_code
        if "```python" in cleaned_code:
            cleaned_code = cleaned_code.split("```python")[1].split("```")[0].strip()
        elif "```" in cleaned_code:
            cleaned_code = cleaned_code.split("```")[1].split("```")[0].strip()

        # 1. Syntax check
        is_syntax_valid = False
        syntax_err = None
        try:
            ast.parse(cleaned_code)
            is_syntax_valid = True
            syntax_valid_count += 1
        except Exception as e:
            syntax_err = str(e)

        # 2. Subprocess execution
        test_script = f"{cleaned_code}\n\n{task['test_assertions']}\n"
        exec_pass = False
        exec_err = None

        if is_syntax_valid:
            try:
                res = subprocess.run(
                    [sys.executable, "-c", test_script],
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                if res.returncode == 0:
                    exec_pass = True
                    assertion_passed_count += 1
                else:
                    exec_err = res.stderr.strip().splitlines()[-1] if res.stderr else "Non-zero exit"
            except subprocess.TimeoutExpired:
                exec_err = "Timeout (2.0s expired)"
            except Exception as e:
                exec_err = str(e)
        else:
            exec_err = f"SyntaxError: {syntax_err}"

        results.append({
            "task_id": task['task_id'],
            "prompt": task['prompt'],
            "generated_code": gen_code,
            "cleaned_code": cleaned_code,
            "syntax_valid": is_syntax_valid,
            "assertion_passed": exec_pass,
            "error": exec_err
        })

        if (idx + 1) % 10 == 0:
            print(f"Task {idx+1:2d}/40 | Syntax Valid: {syntax_valid_count}/{idx+1} | Passed: {assertion_passed_count}/{idx+1}")

    summary = {
        "total_tasks": len(coding_tasks),
        "syntax_valid": syntax_valid_count,
        "syntax_valid_pct": (syntax_valid_count / len(coding_tasks)) * 100,
        "assertion_passed": assertion_passed_count,
        "assertion_passed_pct": (assertion_passed_count / len(coding_tasks)) * 100,
        "details": results
    }

    out_file = Path("data/sealed_eval_015/coding_eval_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Coding Evaluation Summary ===")
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Syntax Validity: {summary['syntax_valid']}/{summary['total_tasks']} ({summary['syntax_valid_pct']:.1f}%)")
    print(f"Assertion Pass Rate: {summary['assertion_passed']}/{summary['total_tasks']} ({summary['assertion_passed_pct']:.1f}%)")
    print(f"Results written to: {out_file}")

if __name__ == '__main__':
    run_coding_evaluation()
