"""
experiments/genesis_stage_j_020/generate_and_eval_1000_benchmark.py
Builds and evaluates a Fresh 1,000-Task Strictly Matched Benchmark:
- 500 In-Distribution tasks
- 500 Out-of-Distribution tasks
- STRICTLY MATCHED operation family distributions between ID and OOD:
  * 24 tasks per operation family across all 20 operations (480 tasks per split)
  * 20 adversarial / unsupported tasks per split
  * Total = 500 ID + 500 OOD = 1,000 tasks
- 100% Unique prompts across all 1,000 tasks
- High-efficiency evaluation with batching, incremental saving after each batch, and resumability.
"""

import sys
import os
import json
import random
import time
import torch
from pathlib import Path

random.seed(20042)
sys.path.insert(0, '.')

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine
from chakrview.runtime.source_grounding import SourceGroundingEngine
from chakrview.runtime.multilingual_normalizer import MultilingualNormalizer
from experiments.genesis_stage_i_019.run_5system_benchmark import DeterministicRuleParser
from experiments.genesis_stage_i_019.generate_nr019_data import generate_sample, generate_adversarial_or_unsupported, OPERATIONS

EXTRA_WORDS = [
    'alpha', 'bravo', 'delta', 'echo', 'foxtrot', 'matrix', 'vector', 'scalar', 'tensor',
    'algorithm', 'recursion', 'cybernetics', 'equilibrium', 'transformer', 'generalization',
    'signal', 'network', 'node', 'graph', 'entropy', 'gradient', 'tensorboard', 'pipeline',
    'benchmark', 'compiler', 'bytecode', 'register', 'kernel', 'scheduler', 'operand',
    'pointer', 'memory', 'buffer', 'socket', 'cluster', 'neuron', 'synapse', 'weight',
    'bias', 'feature', 'layer', 'attention', 'residual', 'embedding', 'encoder', 'decoder',
    'hyperplane', 'manifold', 'stochastic', 'eigenvalue', 'covariance', 'likelihood'
]

EXTRA_VARS = [
    'count', 'total', 'score', 'idx', 'temp', 'counter', 'limit', 'offset', 'pointer',
    'flag', 'status', 'accumulator', 'result', 'target', 'current', 'anchor', 'step',
    'state', 'buffer', 'memory', 'cursor', 'index', 'factor', 'weight', 'bias'
]

def generate_robust_sample(b_id, lang, is_ood, force_op, seed_idx):
    s = generate_sample(b_id, lang, is_ood=is_ood, force_op=force_op)
    target_json = s['target_json']
    target_args = target_json.get('arguments', {})

    if force_op in ['string_upper', 'string_lower', 'string_reverse', 'string_length']:
        w_idx = seed_idx if not is_ood else (seed_idx + 25)
        w = EXTRA_WORDS[w_idx % len(EXTRA_WORDS)]
        w_up = w.upper()
        if force_op == 'string_upper':
            target_args['string'] = w
            s['expected_ans'] = w_up
            if lang == 'en': s['prompt'] = f"Convert '{w}' to uppercase."
            elif lang == 'hi': s['prompt'] = f"शब्द '{w}' को बड़े अक्षरों (uppercase) में बदलें।"
            else: s['prompt'] = f"Word '{w}' ko uppercase me convert karo."
        elif force_op == 'string_lower':
            target_args['string'] = w_up
            s['expected_ans'] = w.lower()
            if lang == 'en': s['prompt'] = f"Convert '{w_up}' to lowercase."
            elif lang == 'hi': s['prompt'] = f"शब्द '{w_up}' को छोटे अक्षरों (lowercase) में लिखें।"
            else: s['prompt'] = f"Word '{w_up}' ko lowercase me badlo."
        elif force_op == 'string_reverse':
            target_args['string'] = w
            s['expected_ans'] = w[::-1]
            if lang == 'en': s['prompt'] = f"Reverse the characters of string '{w}'."
            elif lang == 'hi': s['prompt'] = f"स्ट्रिंग '{w}' के अक्षरों को उल्टा करें।"
            else: s['prompt'] = f"String '{w}' ko reverse karo."
        elif force_op == 'string_length':
            target_args['string'] = w
            s['expected_ans'] = len(w)
            if lang == 'en': s['prompt'] = f"Count the number of characters in '{w}'."
            elif lang == 'hi': s['prompt'] = f"स्ट्रिंग '{w}' में अक्षरों की संख्या गिनें।"
            else: s['prompt'] = f"String '{w}' ki length kitni hai?"
    elif force_op == 'var_set':
        v = EXTRA_VARS[seed_idx % len(EXTRA_VARS)]
        n1 = target_args['value'] + (500 if is_ood else 0)
        target_args['name'] = v
        target_args['value'] = n1
        s['expected_ans'] = f"SET_{v.upper()}"
        if lang == 'en': s['prompt'] = f"Assign value {n1} to variable {v}."
        elif lang == 'hi': s['prompt'] = f"चर (variable) {v} का मान {n1} निर्धारित करें।"
        else: s['prompt'] = f"Variable {v} me value {n1} set kardo."
    elif force_op == 'var_get':
        v = EXTRA_VARS[seed_idx % len(EXTRA_VARS)]
        target_args['name'] = v
        s['expected_ans'] = f"GET_{v.upper()}"
        if lang == 'en': s['prompt'] = f"Retrieve the stored value of variable {v} (slot {seed_idx})."
        elif lang == 'hi': s['prompt'] = f"चर {v} का संचित मान प्राप्त करें (स्लॉट {seed_idx})।"
        else: s['prompt'] = f"Variable {v} ki stored value nikalo (slot {seed_idx})."
    elif force_op == 'stack_push':
        base_val = target_args.get('value', 10)
        n1 = base_val + seed_idx + (1000 if is_ood else 0)
        target_args['value'] = n1
        s['expected_ans'] = f"PUSH_{n1}"
        if lang == 'en': s['prompt'] = f"Push {n1} onto the stack."
        elif lang == 'hi': s['prompt'] = f"स्टैक पर {n1} पुश (push) करें।"
        else: s['prompt'] = f"Stack par value {n1} push kardo."
    elif force_op == 'stack_pop':
        s['expected_ans'] = "POP_DONE"
        if lang == 'en': s['prompt'] = f"Pop the top element from the stack (op #{seed_idx + (50 if is_ood else 0)})."
        elif lang == 'hi': s['prompt'] = f"स्टैक के शीर्ष तत्व को बाहर निकालें (pop करें) (कार्रवाई #{seed_idx + (50 if is_ood else 0)})।"
        else: s['prompt'] = f"Stack se top element pop kardo (action #{seed_idx + (50 if is_ood else 0)})."
    elif force_op == 'stack_peek':
        s['expected_ans'] = "PEEK_DONE"
        if lang == 'en': s['prompt'] = f"Inspect the top of the stack without removing it (step {seed_idx + (50 if is_ood else 0)})."
        elif lang == 'hi': s['prompt'] = f"स्टैक के शीर्ष तत्व को देखें (peek करें) (चरण {seed_idx + (50 if is_ood else 0)})।"
        else: s['prompt'] = f"Stack ka top element peek karo (step #{seed_idx + (50 if is_ood else 0)})."

    s['target_json'] = target_json
    s['target_str'] = json.dumps(target_json)
    return s

def build_1000_matched_benchmark():
    manifest_path = Path("data/stage_j_020/fresh_matched_benchmark_1000.json")
    if manifest_path.exists():
        print(f"Loading existing benchmark manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Building 1,000-Task Matched Benchmark...")
    random.seed(20042)
    bench_tasks = []
    seen_prompts = set()
    b_id = 1

    for is_ood in [False, True]:
        split_name = "OOD" if is_ood else "ID"
        print(f"Generating 500 {split_name} tasks...")
        
        for op in OPERATIONS:
            count = 0
            attempts = 0
            while count < 24 and attempts < 300:
                attempts += 1
                lang = random.choice(["en", "hi", "hinglish"])
                s = generate_robust_sample(f"BMK-{b_id:04d}", lang, is_ood, op, count + attempts * 30)
                if s["prompt"] not in seen_prompts:
                    seen_prompts.add(s["prompt"])
                    bench_tasks.append(s)
                    b_id += 1
                    count += 1
            if count < 24:
                raise RuntimeError(f"Could not generate 24 unique tasks for {op} in {attempts} attempts")

        # 20 adversarial / unsupported tasks per split
        adv_count = 0
        attempts = 0
        while adv_count < 20 and attempts < 100:
            attempts += 1
            lang = random.choice(["en", "hi", "hinglish"])
            s = generate_adversarial_or_unsupported(f"BMK-{b_id:04d}", lang, idx=adv_count + attempts * 25 + (100 if is_ood else 0))
            if s["prompt"] not in seen_prompts:
                seen_prompts.add(s["prompt"])
                bench_tasks.append(s)
                b_id += 1
                adv_count += 1
        if adv_count < 20:
            raise RuntimeError(f"Could not generate 20 unique adversarial tasks in {attempts} attempts")

    print(f"Total benchmark tasks created: {len(bench_tasks)} (Unique: {len(seen_prompts)})")
    os.makedirs("data/stage_j_020", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(bench_tasks, f, indent=2, ensure_ascii=False)
    return bench_tasks

def evaluate_matched_benchmark(diagnostic_only=False, max_tasks=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== EVALUATING MATCHED BENCHMARK ON {device.upper()} ===")

    bench_tasks = build_1000_matched_benchmark()
    if diagnostic_only and max_tasks is None:
        max_tasks = 20
    if max_tasks is not None:
        bench_tasks = bench_tasks[:max_tasks]
        print(f"Running DIAGNOSTIC on first {len(bench_tasks)} tasks")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    executor = RestrictedExecutionEngine()
    rule_parser = DeterministicRuleParser()
    grounding_engine = SourceGroundingEngine()

    staged_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    st_sd = torch.load(staged_path, map_location=device, weights_only=False)
    raw_cfg = st_sd['config']
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)

    # Pure neural model
    model_a = ChakrGenesis17M(cfg).to(device)
    model_a.load_state_dict(st_sd['model_state_dict'])
    model_a.eval()

    # Router model
    router_path = "experiments/genesis_stage_h_018/neural_router_17m.pt"
    st_router = torch.load(router_path, map_location=device, weights_only=False)
    model_router = ChakrGenesis17M(cfg).to(device)
    model_router.load_state_dict(st_router['model_state_dict'])
    model_router.eval()

    systems = ["System_A_Pure_Neural", "System_B_Deterministic_Parser", "System_C_Neural_Router_Direct", "System_D_Neural_Source_Grounded", "System_E_Cascaded_Hybrid"]
    
    results_file = Path("data/stage_j_020/matched_benchmark_1000_results.json" if not diagnostic_only else "data/stage_j_020/diagnostic_20_results.json")
    
    # Check for resumable partial results
    eval_log = []
    completed_task_ids = set()
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                eval_log = prev_data.get("details", [])
                completed_task_ids = {item["task_id"] for item in eval_log}
                print(f"Resuming evaluation: found {len(completed_task_ids)} already completed tasks in {results_file}")
        except Exception as e:
            print(f"Could not load previous results ({e}), starting fresh.")
            eval_log = []
            completed_task_ids = set()

    t0 = time.time()
    batch_size = 10
    total_to_evaluate = len(bench_tasks)

    for idx, task in enumerate(bench_tasks):
        task_id = task["task_id"]
        if task_id in completed_task_ids:
            continue

        prompt = task["prompt"]
        exp_ans = str(task["expected_ans"])
        is_ood = task["is_ood"]
        lang = task["lang"]

        # 1. System A: Pure Neural
        p_ids = torch.tensor([tokenizer.encode(f"<user>\n{prompt}\n<assistant>\n")], dtype=torch.long, device=device)
        curr_a = p_ids
        with torch.no_grad():
            for _ in range(35):
                out, _ = model_a(curr_a)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_a = torch.cat([curr_a, nxt], dim=1)
                if nxt.item() == 1: break
        raw_a = tokenizer.decode(curr_a[0, p_ids.size(1):].tolist(), errors="replace").strip()
        pass_a = (exp_ans == raw_a) or (exp_ans in raw_a)

        # 2. System B: Deterministic Rule Parser
        res_b = rule_parser.parse_and_execute(prompt, executor)
        ans_b = str(res_b.get("result", ""))
        pass_b = (exp_ans == ans_b) or (exp_ans in ans_b)

        # 3. System C: Neural Router Direct
        curr_c = p_ids
        with torch.no_grad():
            for _ in range(65):
                out, _ = model_router(curr_c)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_c = torch.cat([curr_c, nxt], dim=1)
                if nxt.item() == 1: break
        json_c_str = tokenizer.decode(curr_c[0, p_ids.size(1):].tolist(), errors="replace").strip()
        
        detected_op = "none"
        ans_c = "PARSE_ERROR"
        try:
            req_c = json.loads(json_c_str)
            detected_op = req_c.get("operation", "none")
            if req_c.get("status") in ["clarify", "unsupported"]:
                ans_c = req_c.get("status").upper()
            else:
                exec_c = executor.execute(req_c)
                ans_c = str(exec_c.get("result", "")) if exec_c["status"] == "success" else "ERROR"
        except Exception:
            import re
            op_m = re.search(r'"operation":\s*"([^"]+)"', json_c_str)
            if op_m: detected_op = op_m.group(1)
            ans_c = "PARSE_ERROR"
        pass_c = (exp_ans == ans_c) or (exp_ans in ans_c)

        # 4. System D: Neural Source Grounded
        if detected_op == "none" or "unsupported" in json_c_str:
            ans_d = "UNSUPPORTED"
        else:
            grounded_req = grounding_engine.ground_arguments(prompt, detected_op)
            if grounded_req["status"] == "clarify":
                ans_d = "CLARIFY"
            else:
                exec_d = executor.execute({"operation": detected_op, "arguments": grounded_req["arguments"]})
                ans_d = str(exec_d.get("result", "")) if exec_d["status"] == "success" else "ERROR"
        pass_d = (exp_ans == ans_d) or (exp_ans in ans_d)

        # 5. System E: Cascaded Hybrid
        if res_b.get("status") == "success":
            ans_e = ans_b
            pass_e = pass_b
        else:
            ans_e = ans_d
            pass_e = pass_d

        passes = [pass_a, pass_b, pass_c, pass_d, pass_e]

        eval_log.append({
            "task_id": task_id,
            "prompt": prompt,
            "expected": exp_ans,
            "is_ood": is_ood,
            "lang": lang,
            "op": task.get("expected_op", "unknown"),
            "detected_op": detected_op,
            "answers": {
                "System_A": raw_a,
                "System_B": ans_b,
                "System_C": ans_c,
                "System_D": ans_d,
                "System_E": ans_e
            },
            "passes": {systems[i]: passes[i] for i in range(len(systems))}
        })
        completed_task_ids.add(task_id)

        # Incremental saving & reporting every batch
        if (len(eval_log) % batch_size == 0) or (idx + 1 == total_to_evaluate):
            elapsed = time.time() - t0
            rate = len(completed_task_ids) / (elapsed + 1e-5)
            remaining_tasks = total_to_evaluate - len(completed_task_ids)
            eta_s = remaining_tasks / (rate + 1e-5)
            print(f"  [Progress] Evaluated {len(completed_task_ids)}/{total_to_evaluate} tasks | "
                  f"Rate: {rate:.2f} task/s | Elapsed: {elapsed:.1f}s | ETA: {eta_s:.1f}s", flush=True)

            # Build interim summary
            summary = _compute_summary(eval_log, systems)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": eval_log}, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print(f"  BENCHMARK EVALUATION COMPLETE ({len(eval_log)} tasks)")
    print("="*80)
    final_summary = _compute_summary(eval_log, systems)
    for s_name in systems:
        s_data = final_summary[s_name]
        print(f"{s_name:32s} | ID: {s_data['id_acc']:5.1f}% | OOD: {s_data['ood_acc']:5.1f}% | Overall: {s_data['overall_acc']:5.1f}% ({s_data['correct']}/{s_data['total']})")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"summary": final_summary, "details": eval_log}, f, indent=2, ensure_ascii=False)

    return final_summary

def _compute_summary(eval_log, systems):
    summary = {}
    tot = len(eval_log)
    tot_id = sum(1 for item in eval_log if not item["is_ood"])
    tot_ood = sum(1 for item in eval_log if item["is_ood"])

    for s in systems:
        correct = sum(1 for item in eval_log if item["passes"][s])
        id_correct = sum(1 for item in eval_log if (not item["is_ood"] and item["passes"][s]))
        ood_correct = sum(1 for item in eval_log if (item["is_ood"] and item["passes"][s]))
        langs = {"en": 0, "hi": 0, "hinglish": 0}
        for item in eval_log:
            if item["passes"][s]:
                l = item.get("lang", "en")
                langs[l] = langs.get(l, 0) + 1

        summary[s] = {
            "total": tot,
            "correct": correct,
            "overall_acc": round((correct / tot * 100), 2) if tot > 0 else 0.0,
            "id_total": tot_id,
            "id_correct": id_correct,
            "id_acc": round((id_correct / tot_id * 100), 2) if tot_id > 0 else 0.0,
            "ood_total": tot_ood,
            "ood_correct": ood_correct,
            "ood_acc": round((ood_correct / tot_ood * 100), 2) if tot_ood > 0 else 0.0,
            "langs": langs
        }
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true", help="Run 20-task diagnostic batch")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum tasks to evaluate")
    args = parser.parse_args()

    evaluate_matched_benchmark(diagnostic_only=args.diagnostic, max_tasks=args.max_tasks)
