"""
scripts/run_m3_evaluations.py

Comprehensive M3 evaluation runner:
1. 60-task Conversational Benchmark (evaluation/conversational_suite/m3_conversational_tasks.json)
2. 32-task Intelligence Benchmark (evaluation/intelligence_suite/intelligence_tasks.json)
3. 12-task Coding Benchmark (evaluation/coding_suite/coding_tasks.json)
4. 30-prompt Neural Benchmark Harness (evaluation/neural_benchmark_harness.py)

Evaluates checkpoints:
- Canonical Step 10,000 (Baseline)
- M2 Arm A Step 50
- M3 Step 50
- M3 Step 100
- M3 Step 250
- M3 Step 500
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.sampling import Sampler, SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from evaluation.intelligence_suite.scoring_evaluator import evaluate_response
from evaluation.coding_suite.run_coding_evaluation import check_python_syntax
from evaluation.coding_suite.isolated_executor import run_process_isolated_test
from evaluation.neural_benchmark_harness import NeuralBenchmarkHarness, save_benchmark_results

def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

CHECKPOINTS = {
    "canonical_step_10000": {
        "label": "Canonical Step 10,000 Baseline",
        "path": REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
    },
    "m2_arm_a_step_50": {
        "label": "M2 Arm A Step 50 Checkpoint",
        "path": REPO_ROOT / "experiments" / "exp34_attractor_suppression" / "arm_a_masked" / "sft_checkpoint_step_50.pt",
    },
    "m3_step_50": {
        "label": "M3 Step 50 Checkpoint",
        "path": REPO_ROOT / "experiments" / "m3_conversational_intelligence" / "runs" / "m3_sft_500steps" / "sft_checkpoint_step_50.pt",
    },
    "m3_step_100": {
        "label": "M3 Step 100 Checkpoint",
        "path": REPO_ROOT / "experiments" / "m3_conversational_intelligence" / "runs" / "m3_sft_500steps" / "sft_checkpoint_step_100.pt",
    },
    "m3_step_250": {
        "label": "M3 Step 250 Checkpoint",
        "path": REPO_ROOT / "experiments" / "m3_conversational_intelligence" / "runs" / "m3_sft_500steps" / "sft_checkpoint_step_250.pt",
    },
    "m3_step_500": {
        "label": "M3 Step 500 Checkpoint",
        "path": REPO_ROOT / "experiments" / "m3_conversational_intelligence" / "runs" / "m3_sft_500steps" / "sft_checkpoint_step_500.pt",
    }
}

OUT_BASE = REPO_ROOT / "experiments" / "m3_conversational_intelligence" / "evaluations"
OUT_BASE.mkdir(parents=True, exist_ok=True)

class DirectNeuralRunner:
    """Direct model generation engine bypassing pipeline HTTP/tenant wrappers while preserving exact tokenizer & model contract."""
    def __init__(self, ckpt_path: Path):
        self.tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
        self.model = ChakrMicro(ModelConfig())
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.sampler = Sampler()

    def generate(self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        curr_tokens = list(prompt_tokens)
        gen_tokens = []
        stop_reason = "MAX_TOKENS"
        sampling_cfg = SamplingConfig(temperature=temperature, top_k=0)

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_in = torch.tensor([curr_tokens], dtype=torch.long)
                logits = self.model(t_in)[0, -1, :]
                next_tok = self.sampler.sample(logits, generated_tokens=gen_tokens, config=sampling_cfg, step=step)
                gen_tokens.append(next_tok)
                curr_tokens.append(next_tok)
                if next_tok == 1: # EOS
                    stop_reason = "STOP_TOKEN"
                    break

        duration_ms = (time.perf_counter() - t0) * 1000.0
        gen_text = self.tokenizer.decode(gen_tokens, errors="replace")
        return {
            "text": gen_text,
            "tokens": gen_tokens,
            "token_count": len(gen_tokens),
            "stop_reason": stop_reason,
            "duration_ms": duration_ms
        }

# 1. Conversational Suite (60 tasks)
def evaluate_conversational(runner: DirectNeuralRunner, ckpt_key: str, ckpt_info: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n--- Running 60-Task Conversational Suite on {ckpt_info['label']} ---")
    tasks_file = REPO_ROOT / "evaluation" / "conversational_suite" / "m3_conversational_tasks.json"
    with open(tasks_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    tasks = bench_data["tasks"]
    
    results = []
    passed_count = 0
    cat_counts = {}
    
    for idx, task in enumerate(tasks):
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        target = task.get("target", "")
        expected = task.get("expected_concept", "")
        scoring_rule = task.get("scoring", "substring")
        max_toks = task.get("max_tokens", 36)
        
        resp = runner.generate(prompt=prompt, max_new_tokens=max_toks, temperature=0.0)
        gen_text = resp["text"]
        stop_reason = resp["stop_reason"]
        
        passed, reason = evaluate_response(
            generated_text=gen_text,
            expected_concept=expected,
            scoring_rule=scoring_rule,
            test_code="",
            full_prompt=prompt
        )
        
        if passed:
            passed_count += 1
            
        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "passed": 0}
        cat_counts[cat]["total"] += 1
        if passed:
            cat_counts[cat]["passed"] += 1
            
        results.append({
            "id": t_id,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "expected_concept": expected,
            "scoring_rule": scoring_rule,
            "generated_text": gen_text,
            "passed": passed,
            "reason": reason,
            "stop_reason": stop_reason,
            "tokens_generated": resp["token_count"],
            "duration_ms": round(resp["duration_ms"], 2)
        })

    summary = {
        "suite_name": "M3_CONVERSATIONAL_BENCHMARK_v1.0",
        "checkpoint_id": ckpt_key,
        "checkpoint_label": ckpt_info["label"],
        "checkpoint_sha256": file_sha256(ckpt_info["path"]),
        "total_tasks": len(tasks),
        "passed_tasks": passed_count,
        "accuracy": round((passed_count / len(tasks)) * 100.0, 2),
        "category_breakdown": {
            c: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": round((v["passed"] / v["total"]) * 100.0, 2)
            } for c, v in cat_counts.items()
        },
        "tasks": results
    }
    
    out_p = OUT_BASE / f"conversational_{ckpt_key}.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Passed: {passed_count}/{len(tasks)} ({summary['accuracy']}%) saved to {out_p.name}")
    return summary

# 2. Intelligence Suite (32 tasks)
def evaluate_intelligence(runner: DirectNeuralRunner, ckpt_key: str, ckpt_info: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n--- Running 32-Task Intelligence Regression Suite on {ckpt_info['label']} ---")
    tasks_file = REPO_ROOT / "evaluation" / "intelligence_suite" / "intelligence_tasks.json"
    with open(tasks_file, "r", encoding="utf-8-sig") as f:
        bench_data = json.load(f)
    tasks = bench_data["tasks"]
    categories = bench_data["categories"]
    
    results = []
    passed_count = 0
    attractor_count = 0
    cat_tallies = {cat: {"total": 0, "passed": 0, "attractor_collapse": 0} for cat in categories}
    
    for idx, task in enumerate(tasks):
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        expected = task.get("expected_concept", "")
        test_code = task.get("test_code", "")
        scoring = task.get("scoring", "substring")
        
        cat_tallies[cat]["total"] += 1
        
        resp = runner.generate(prompt=prompt, max_new_tokens=32, temperature=0.0)
        gen_text = resp["text"]
        
        words = gen_text.strip().lower().split()
        is_attractor = False
        if len(words) >= 6:
            tri_grams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            if len(tri_grams) > len(set(tri_grams)) + 1:
                is_attractor = True
                attractor_count += 1
                cat_tallies[cat]["attractor_collapse"] += 1
                
        passed, score_reason = evaluate_response(
            generated_text=gen_text,
            expected_concept=expected,
            scoring_rule=scoring,
            test_code=test_code,
            full_prompt=prompt
        )
        
        if passed:
            passed_count += 1
            cat_tallies[cat]["passed"] += 1
            
        results.append({
            "id": t_id,
            "category": cat,
            "prompt": prompt,
            "expected_concept": expected,
            "generated_text": gen_text,
            "scoring_rule": scoring,
            "passed": passed,
            "score_reason": score_reason,
            "attractor_collapse": is_attractor,
            "stop_reason": resp["stop_reason"],
            "token_count": resp["token_count"],
            "latency_ms": round(resp["duration_ms"], 2)
        })

    summary = {
        "suite_name": bench_data.get("suite_name", "CHAKRVIEW_INTELLIGENCE_BENCHMARK"),
        "checkpoint_id": ckpt_key,
        "checkpoint_label": ckpt_info["label"],
        "checkpoint_sha256": file_sha256(ckpt_info["path"]),
        "total_tasks": len(tasks),
        "passed_tasks_count": passed_count,
        "attractor_collapse_count": attractor_count,
        "overall_pass_rate": f"{passed_count}/{len(tasks)} ({round((passed_count/len(tasks))*100, 1)}%)",
        "attractor_rate": f"{attractor_count}/{len(tasks)} ({round((attractor_count/len(tasks))*100, 1)}%)",
        "category_metrics": {
            cat: {
                "total": t["total"],
                "passed": t["passed"],
                "pass_rate": f"{t['passed']}/{t['total']} ({round((t['passed']/t['total'])*100, 1)}%)" if t['total'] > 0 else "0/0",
                "attractor_collapse_count": t["attractor_collapse"]
            } for cat, t in cat_tallies.items()
        },
        "task_results": results
    }
    
    out_p = OUT_BASE / f"intelligence_{ckpt_key}.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Passed: {passed_count}/{len(tasks)} ({summary['overall_pass_rate']}) | Attractors: {summary['attractor_rate']} saved to {out_p.name}")
    return summary

# 3. Coding Suite (12 tasks)
def evaluate_coding(runner: DirectNeuralRunner, ckpt_key: str, ckpt_info: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n--- Running 12-Task Coding Suite on {ckpt_info['label']} ---")
    tasks_file = REPO_ROOT / "evaluation" / "coding_suite" / "coding_tasks.json"
    with open(tasks_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    tasks = bench_data["tasks"]
    
    results = []
    syntax_count = 0
    test_passed_count = 0
    match_count = 0
    timeout_count = 0
    policy_rejected_count = 0
    empty_count = 0
    
    for idx, task in enumerate(tasks):
        task_id = task["id"]
        category = task["category"]
        prompt = task["prompt"]
        target = task.get("target_snippet", "")
        val_type = task.get("validation_type", "unit_test")
        test_code = task.get("test_code", "")
        
        resp = runner.generate(prompt=prompt, max_new_tokens=48, temperature=0.0)
        gen_text = resp["text"]
        gen_duration_ms = resp["duration_ms"]
        
        is_empty = len(gen_text.strip()) == 0
        if is_empty:
            empty_count += 1
            
        full_code = prompt + gen_text
        is_syntax = check_python_syntax(full_code)
        if is_syntax:
            syntax_count += 1
            
        test_passed = False
        substring_matched = False
        exact_matched = gen_text.strip() == target.strip()
        exec_err = None
        exec_latency = 0.0
        timed_out = False
        rejected_by_policy = False
        
        clean_gen = gen_text.strip()
        lines = [l.strip() for l in clean_gen.splitlines() if l.strip()]
        non_comment_lines = [l for l in lines if not l.startswith("#")]
        has_executable_code = len(non_comment_lines) > 0

        if val_type == "unit_test" and test_code:
            if not has_executable_code:
                test_passed = False
                exec_err = "NO_EXECUTABLE_CODE: completion is empty or comments only"
            else:
                exec_res = run_process_isolated_test(full_code, test_code, timeout_seconds=2.0)
                test_passed = exec_res["passed"]
                exec_err = exec_res["error"]
                exec_latency = exec_res.get("latency_ms", 0.0)
                timed_out = exec_res.get("timed_out", False)
                rejected_by_policy = exec_res.get("rejected_by_policy", False)
                if test_passed:
                    test_passed_count += 1
                if timed_out:
                    timeout_count += 1
                if rejected_by_policy:
                    policy_rejected_count += 1
        elif val_type == "substring_match":
            substring_matched = target.lower() in gen_text.lower() if gen_text else False
            if substring_matched:
                match_count += 1
                test_passed = True
                test_passed_count += 1
        elif val_type == "exact_match":
            test_passed = exact_matched
            if exact_matched:
                match_count += 1
                test_passed_count += 1
                
        results.append({
            "id": task_id,
            "category": category,
            "title": task.get("title", task_id),
            "prompt": prompt,
            "generated_text": gen_text,
            "target_snippet": target,
            "syntax_valid": is_syntax,
            "test_passed": test_passed,
            "exact_match": exact_matched,
            "substring_match": substring_matched,
            "empty_output": is_empty,
            "timed_out": timed_out,
            "rejected_by_policy": rejected_by_policy,
            "error": exec_err,
            "gen_latency_ms": round(gen_duration_ms, 2),
            "exec_latency_ms": exec_latency
        })
        
    summary = {
        "suite_name": bench_data.get("suite_name", "CHAKRVIEW_CODING_BENCHMARK_v0.7"),
        "checkpoint_id": ckpt_key,
        "checkpoint_label": ckpt_info["label"],
        "checkpoint_sha256": file_sha256(ckpt_info["path"]),
        "total_tasks": len(tasks),
        "syntax_valid_count": syntax_count,
        "test_passed_count": test_passed_count,
        "match_count": match_count,
        "timeout_count": timeout_count,
        "empty_output_count": empty_count,
        "policy_rejected_count": policy_rejected_count,
        "syntax_valid_rate": f"{syntax_count}/{len(tasks)} ({round(syntax_count/len(tasks)*100, 1)}%)",
        "overall_success_rate": f"{test_passed_count}/{len(tasks)} ({round(test_passed_count/len(tasks)*100, 1)}%)",
        "task_results": results
    }
    
    out_p = OUT_BASE / f"coding_{ckpt_key}.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Passed: {test_passed_count}/{len(tasks)} | Syntax: {summary['syntax_valid_rate']} saved to {out_p.name}")
    return summary

# 4. Neural Benchmark Harness (30 prompts)
def evaluate_neural_harness(ckpt_key: str, ckpt_info: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n--- Running 30-Prompt Neural Benchmark Harness on {ckpt_info['label']} ---")
    harness = NeuralBenchmarkHarness(
        checkpoint_label=ckpt_info["label"],
        checkpoint_path=ckpt_info["path"],
        tokenizer_dir=REPO_ROOT / "data" / "experiments" / "vocab_4096",
        cpu_threads=2,
        seed=42
    )
    
    prompts_path = REPO_ROOT / "evaluation" / "benchmark_prompts_v2.0.json"
    report = harness.run_benchmark(
        prompts_path=prompts_path,
        max_new_tokens=32,
        temperature=0.0
    )
    
    out_json = OUT_BASE / f"neural_harness_{ckpt_key}.json"
    out_md = OUT_BASE / f"neural_harness_{ckpt_key}.md"
    save_benchmark_results(report, out_json, out_md)
    
    with open(out_json, "r", encoding="utf-8") as f:
        h_data = json.load(f)
        
    print(f"Harness: 4-gram rep={h_data['mean_repeated_4gram_ratio']:.4f}, unique_tok={h_data['mean_unique_token_ratio']:.4f}, EOS count={h_data['eos_termination_count']}/{h_data['total_prompts']}")
    return {
        "mean_repeated_4gram_ratio": h_data["mean_repeated_4gram_ratio"],
        "mean_unique_token_ratio": h_data["mean_unique_token_ratio"],
        "eos_termination_count": h_data["eos_termination_count"],
        "total_prompts": h_data["total_prompts"],
        "mean_throughput_tps": h_data["mean_throughput_tps"],
        "deterministic_accuracy": h_data["deterministic_accuracy"]
    }

def main():
    print("=================================================================")
    print("M3 UNIFIED MULTI-CHECKPOINT BENCHMARK SUITE")
    print("=================================================================")
    
    master_summary = {}
    
    for ckpt_key, ckpt_info in CHECKPOINTS.items():
        print(f"\n=================================================================")
        print(f"STARTING COMPREHENSIVE SUITE FOR: {ckpt_key}")
        print(f"Label: {ckpt_info['label']}")
        print(f"Path:  {ckpt_info['path']}")
        print(f"SHA-256: {file_sha256(ckpt_info['path'])}")
        print(f"=================================================================")
        
        # Build direct runner
        runner = DirectNeuralRunner(ckpt_info["path"])
        
        # 1. Conversational Suite (60 tasks)
        conv_res = evaluate_conversational(runner, ckpt_key, ckpt_info)
        
        # 2. Intelligence Suite (32 tasks)
        intel_res = evaluate_intelligence(runner, ckpt_key, ckpt_info)
        
        # 3. Coding Suite (12 tasks)
        coding_res = evaluate_coding(runner, ckpt_key, ckpt_info)
        
        # 4. Neural Harness (30 prompts)
        harness_res = evaluate_neural_harness(ckpt_key, ckpt_info)
        
        master_summary[ckpt_key] = {
            "label": ckpt_info["label"],
            "sha256": file_sha256(ckpt_info["path"]),
            "conversational_60": {
                "passed": conv_res["passed_tasks"],
                "total": conv_res["total_tasks"],
                "accuracy": conv_res["accuracy"],
                "breakdown": conv_res["category_breakdown"]
            },
            "intelligence_32": {
                "passed": intel_res["passed_tasks_count"],
                "total": intel_res["total_tasks"],
                "overall_pass_rate": intel_res["overall_pass_rate"],
                "attractor_rate": intel_res["attractor_rate"]
            },
            "coding_12": {
                "tests_passed": coding_res["test_passed_count"],
                "syntax_valid": coding_res["syntax_valid_count"],
                "total": coding_res["total_tasks"],
                "overall_success_rate": coding_res["overall_success_rate"],
                "syntax_valid_rate": coding_res["syntax_valid_rate"]
            },
            "neural_harness_30": harness_res
        }
        
    master_json_p = OUT_BASE / "m3_comprehensive_evaluation_matrix.json"
    with open(master_json_p, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
        
    print("\n=================================================================")
    print("ALL 6 CHECKPOINTS EVALUATED ACROSS ALL 4 BENCHMARKS!")
    print(f"Master evaluation matrix written to: {master_json_p}")
    print("=================================================================")

if __name__ == "__main__":
    main()
