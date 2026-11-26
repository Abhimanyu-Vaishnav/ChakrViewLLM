"""
experiments/genesis_stage_i_019/run_5system_benchmark.py
Evaluates 5 Systems on the 800-Task Sealed Benchmark:
- System A: Pure Neural Baseline (ChakrGenesis-17M Stage-D direct generation)
- System B: Deterministic Pattern Parser + Restricted Executor
- System C: Neural Router Direct (Free-form JSON generation)
- System D: Neural Router + Source-Span Grounding + Multilingual Normalizer
- System E: Cascaded Hybrid Router (Deterministic Parser -> Neural Grounding Fallback)
"""

import sys
import os
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine
from chakrview.runtime.source_grounding import SourceGroundingEngine
from chakrview.runtime.multilingual_normalizer import MultilingualNormalizer

class DeterministicRuleParser:
    """NR-018 Regex Rule-based Baseline"""
    def __init__(self):
        import re
        self.re = re

    def parse_and_execute(self, prompt: str, executor: RestrictedExecutionEngine) -> dict:
        p = prompt.strip()
        # Unsupported check
        if any(w in p.lower() for w in ["os.system", "eval(", "rm -rf", "cmd.exe", "whoami"]):
            return {"status": "unsupported", "result": "UNSUPPORTED"}

        # Math Mul
        m_mul = self.re.search(r"Multiply\s+(\d+)\s+by\s+(\d+)", p, self.re.I)
        if m_mul:
            return executor.execute({"operation": "math_mul", "arguments": {"a": int(m_mul.group(1)), "b": int(m_mul.group(2))}})
        # Math Add
        m_add = self.re.search(r"Add\s+(\d+)\s+to\s+(\d+)|Calculate the sum of\s+(\d+)\s+and\s+(\d+)", p, self.re.I)
        if m_add:
            a = int(m_add.group(1) or m_add.group(3))
            b = int(m_add.group(2) or m_add.group(4))
            return executor.execute({"operation": "math_add", "arguments": {"a": a, "b": b}})
        # Math Sub
        m_sub = self.re.search(r"Subtract\s+(\d+)\s+from\s+(\d+)", p, self.re.I)
        if m_sub:
            return executor.execute({"operation": "math_sub", "arguments": {"a": int(m_sub.group(2)), "b": int(m_sub.group(1))}})
        # Math Div
        m_div = self.re.search(r"Divide\s+(\d+)\s+by\s+(\d+)", p, self.re.I)
        if m_div:
            return executor.execute({"operation": "math_div", "arguments": {"a": int(m_div.group(1)), "b": int(m_div.group(2))}})

        # List reverse
        m_lrev = self.re.search(r"Reverse the list\s+\[([0-9,\s]+)\]", p, self.re.I)
        if m_lrev:
            lst = [int(x.strip()) for x in m_lrev.group(1).split(",") if x.strip()]
            return executor.execute({"operation": "list_reverse", "arguments": {"list": lst}})

        # String reverse
        m_srev = self.re.search(r"Reverse the characters of\s+['\"]([^'\"]+)['\"]", p, self.re.I)
        if m_srev:
            return executor.execute({"operation": "string_reverse", "arguments": {"string": m_srev.group(1)}})

        return {"status": "unsupported", "result": "UNSUPPORTED"}

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NR-019 5-SYSTEM SEALED BENCHMARK EVALUATION ON {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    executor = RestrictedExecutionEngine()
    rule_parser = DeterministicRuleParser()
    grounding_engine = SourceGroundingEngine()

    with open("data/stage_i_019/sealed_benchmark_800.json", "r", encoding="utf-8") as f:
        bench_tasks = json.load(f)

    # 1. Load Stage-D Checkpoint (System A)
    staged_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    st_sd = torch.load(staged_path, map_location=device, weights_only=False)
    raw_cfg = st_sd['config']
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)

    model_a = ChakrGenesis17M(cfg).to(device)
    model_a.load_state_dict(st_sd['model_state_dict'])
    model_a.eval()

    # 2. Load Neural Router Checkpoint (System C, D, E)
    router_path = "experiments/genesis_stage_h_018/neural_router_17m.pt"
    st_router = torch.load(router_path, map_location=device, weights_only=False)
    model_router = ChakrGenesis17M(cfg).to(device)
    model_router.load_state_dict(st_router['model_state_dict'])
    model_router.eval()

    metrics = {
        "System_A_Pure_Neural": {"correct": 0, "total": 0, "in_correct": 0, "in_tot": 0, "ood_correct": 0, "ood_tot": 0, "lang": {"en":0, "hi":0, "hinglish":0}},
        "System_B_Deterministic_Parser": {"correct": 0, "total": 0, "in_correct": 0, "in_tot": 0, "ood_correct": 0, "ood_tot": 0, "lang": {"en":0, "hi":0, "hinglish":0}},
        "System_C_Neural_Router_Direct": {"correct": 0, "total": 0, "in_correct": 0, "in_tot": 0, "ood_correct": 0, "ood_tot": 0, "lang": {"en":0, "hi":0, "hinglish":0}},
        "System_D_Neural_Source_Grounded": {"correct": 0, "total": 0, "in_correct": 0, "in_tot": 0, "ood_correct": 0, "ood_tot": 0, "lang": {"en":0, "hi":0, "hinglish":0}},
        "System_E_Cascaded_Hybrid": {"correct": 0, "total": 0, "in_correct": 0, "in_tot": 0, "ood_correct": 0, "ood_tot": 0, "lang": {"en":0, "hi":0, "hinglish":0}}
    }

    detailed_log = []
    print("Evaluating 800 tasks...")
    t0 = time.time()

    for idx, task in enumerate(bench_tasks):
        prompt = task['prompt']
        exp_ans = str(task['expected_ans'])
        is_ood = task['is_ood']
        lang = task['lang']

        # === System A: Pure Neural ===
        p_ids = torch.tensor([tokenizer.encode(f"<user>\n{prompt}\n<assistant>\n")], dtype=torch.long, device=device)
        curr_a = p_ids
        with torch.no_grad():
            for _ in range(40):
                out, _ = model_a(curr_a)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_a = torch.cat([curr_a, nxt], dim=1)
                if nxt.item() == 1: break
        raw_a = tokenizer.decode(curr_a[0, p_ids.size(1):].tolist()).strip()
        pass_a = (exp_ans == raw_a) or (exp_ans in raw_a)

        # === System B: Deterministic Pattern Parser ===
        res_b = rule_parser.parse_and_execute(prompt, executor)
        ans_b = str(res_b.get("result", ""))
        pass_b = (exp_ans == ans_b) or (exp_ans in ans_b)

        # === System C: Neural Router Direct (Free-form JSON) ===
        curr_c = p_ids
        with torch.no_grad():
            for _ in range(70):
                out, _ = model_router(curr_c)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_c = torch.cat([curr_c, nxt], dim=1)
                if nxt.item() == 1: break
        json_c_str = tokenizer.decode(curr_c[0, p_ids.size(1):].tolist()).strip()
        
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
            # Try to recover operation name via regex if json truncated
            import re
            op_m = re.search(r'"operation":\s*"([^"]+)"', json_c_str)
            if op_m: detected_op = op_m.group(1)
            ans_c = "PARSE_ERROR"
        pass_c = (exp_ans == ans_c) or (exp_ans in ans_c)

        # === System D: Neural Router + Source-Span Grounding ===
        # Use detected operation from router, but ground arguments strictly from source text!
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

        # === System E: Cascaded Hybrid Router ===
        # 1. Try deterministic pattern parser first
        # 2. If unsupported/unparsed, fallback to System D (Neural Intent + Source Grounding)
        if res_b.get("status") == "success":
            ans_e = ans_b
            pass_e = pass_b
        else:
            ans_e = ans_d
            pass_e = pass_d

        # Record metrics
        for s_name, p_flag in [
            ("System_A_Pure_Neural", pass_a),
            ("System_B_Deterministic_Parser", pass_b),
            ("System_C_Neural_Router_Direct", pass_c),
            ("System_D_Neural_Source_Grounded", pass_d),
            ("System_E_Cascaded_Hybrid", pass_e)
        ]:
            metrics[s_name]["total"] += 1
            if p_flag:
                metrics[s_name]["correct"] += 1
                metrics[s_name]["lang"][lang] = metrics[s_name]["lang"].get(lang, 0) + 1
            if is_ood:
                metrics[s_name]["ood_tot"] += 1
                if p_flag: metrics[s_name]["ood_correct"] += 1
            else:
                metrics[s_name]["in_tot"] += 1
                if p_flag: metrics[s_name]["in_correct"] += 1

        detailed_log.append({
            "task_id": task['task_id'],
            "prompt": prompt,
            "expected": exp_ans,
            "is_ood": is_ood,
            "lang": lang,
            "sys_a": {"out": raw_a, "pass": pass_a},
            "sys_b": {"out": ans_b, "pass": pass_b},
            "sys_c": {"out": ans_c, "pass": pass_c},
            "sys_d": {"out": ans_d, "pass": pass_d},
            "sys_e": {"out": ans_e, "pass": pass_e}
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/800 tasks... (Elapsed: {time.time()-t0:.1f}s)")

    print("\n" + "="*80)
    print("  NR-019 5-SYSTEM SEALED BENCHMARK RESULTS (800 TASKS)")
    print("="*80)
    summary_report = {}
    for s_name, m in metrics.items():
        in_acc = (m['in_correct'] / m['in_tot']) * 100 if m['in_tot'] else 0.0
        ood_acc = (m['ood_correct'] / m['ood_tot']) * 100 if m['ood_tot'] else 0.0
        overall = (m['correct'] / m['total']) * 100
        summary_report[s_name] = {
            "overall_acc": round(overall, 2),
            "in_dist_acc": round(in_acc, 2),
            "ood_acc": round(ood_acc, 2),
            "correct": m['correct'],
            "total": m['total'],
            "lang_acc": {k: round((v / (200 if k != 'mixed' else 200)) * 100, 2) for k, v in m['lang'].items()}
        }
        print(f"{s_name:32s} | In: {in_acc:5.1f}% | OOD: {ood_acc:5.1f}% | Overall: {overall:5.1f}% ({m['correct']}/{m['total']})")

    with open("data/stage_i_019/benchmark_800_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary_report, "details": detailed_log}, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to data/stage_i_019/benchmark_800_results.json")

if __name__ == "__main__":
    run_benchmark()
