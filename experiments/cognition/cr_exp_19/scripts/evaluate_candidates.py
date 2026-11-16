"""
evaluate_candidates.py - Comprehensive Evaluation of CR-19 Checkpoints on Frozen Benchmarks
Evaluates:
- Baseline (CR-17 conversational parent)
- Arm A (Short-Horizon Control: seeds 101, 202, 303)
- Arm B (Extended-Horizon Treatment: seeds 101, 202, 303)

Runs both:
- CR19_CONTEXT_200.json (All 200 multi-turn context probes across 1, 2, 3, 5, 8 turns)
- CR19_CONVERSATION_100.json (All 100 conversational quality probes across greeting, identity, hinglish, dialogue)
"""
import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig
from chakrview.chat import ConversationalChatCLI

BASE_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_19"
BENCHMARK_DIR = BASE_DIR / "benchmark"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CKPT = REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"

def evaluate_model_on_context(runner, cli, probes, cfg):
    """
    Evaluates on CR19_CONTEXT_200.json using both Direct Prompt (Path A) and CLI Runtime packing (Path B).
    """
    tok = runner.tokenizer
    by_group = {"1-turn": {"total": 0, "hits_a": 0, "hits_b": 0},
                "2-turn": {"total": 0, "hits_a": 0, "hits_b": 0},
                "3-turn": {"total": 0, "hits_a": 0, "hits_b": 0},
                "5-turn": {"total": 0, "hits_a": 0, "hits_b": 0},
                "8-turn": {"total": 0, "hits_a": 0, "hits_b": 0}}
    by_task_type = {}
    detailed_results = []

    for p in probes:
        pid = p["id"]
        group = p["group"]
        ttype = p.get("task_type", "unknown")
        exp = p["expected"]
        prompt_full = p["formatted_prompt"]

        if ttype not in by_task_type:
            by_task_type[ttype] = {"total": 0, "hits_a": 0, "hits_b": 0}

        # Path A: Direct model inference with formatted prompt
        toks_a = tok.encode(prompt_full, add_bos=True, add_eos=False)
        curr_a = list(toks_a)
        gen_a = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr_a], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen_a, config=cfg, step=step)
                gen_a.append(tok_id)
                curr_a.append(tok_id)
                if tok_id == 1:
                    break
        resp_a = tok.decode(gen_a, errors="replace").strip()
        hit_a = any(e.lower() in resp_a.lower() for e in exp)

        # Path B: CLI Runtime prompt packing
        cli.history = []
        for t in p["turns"][:-1]:
            cli.history.append({"user": t["user"], "assistant": t["assistant"]})
        last_user = p["turns"][-1]["user"]
        p_b, _ = cli.build_prompt(last_user)

        toks_b = tok.encode(p_b, add_bos=True, add_eos=False)
        curr_b = list(toks_b)
        gen_b = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr_b], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen_b, config=cfg, step=step)
                gen_b.append(tok_id)
                curr_b.append(tok_id)
                if tok_id == 1:
                    break
        resp_b = tok.decode(gen_b, errors="replace").strip()
        hit_b = any(e.lower() in resp_b.lower() for e in exp)

        by_group[group]["total"] += 1
        by_task_type[ttype]["total"] += 1
        if hit_a:
            by_group[group]["hits_a"] += 1
            by_task_type[ttype]["hits_a"] += 1
        if hit_b:
            by_group[group]["hits_b"] += 1
            by_task_type[ttype]["hits_b"] += 1

        detailed_results.append({
            "id": pid,
            "group": group,
            "task_type": ttype,
            "expected": exp,
            "resp_direct": resp_a,
            "hit_direct": hit_a,
            "resp_cli": resp_b,
            "hit_cli": hit_b
        })

    summary = {
        "total_probes": len(probes),
        "total_hits_direct": sum(v["hits_a"] for v in by_group.values()),
        "accuracy_direct": sum(v["hits_a"] for v in by_group.values()) / max(1, len(probes)),
        "total_hits_cli": sum(v["hits_b"] for v in by_group.values()),
        "accuracy_cli": sum(v["hits_b"] for v in by_group.values()) / max(1, len(probes)),
        "by_group": {
            k: {
                "total": v["total"],
                "hits_direct": v["hits_a"],
                "acc_direct": v["hits_a"] / max(1, v["total"]),
                "hits_cli": v["hits_b"],
                "acc_cli": v["hits_b"] / max(1, v["total"])
            }
            for k, v in by_group.items()
        },
        "by_task_type": {
            k: {
                "total": v["total"],
                "hits_direct": v["hits_a"],
                "acc_direct": v["hits_a"] / max(1, v["total"]),
                "hits_cli": v["hits_b"],
                "acc_cli": v["hits_b"] / max(1, v["total"])
            }
            for k, v in by_task_type.items()
        }
    }
    return summary, detailed_results

def evaluate_model_on_conversation(runner, probes, cfg):
    """
    Evaluates on CR19_CONVERSATION_100.json.
    """
    tok = runner.tokenizer
    by_category = {}
    detailed_results = []

    for p in probes:
        pid = p["id"]
        cat = p["category"]
        prompt = p["prompt"]
        exp = p["expected"]

        if cat not in by_category:
            by_category[cat] = {"total": 0, "hits": 0}

        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    break
        resp = tok.decode(gen, errors="replace").strip()
        hit = any(e.lower() in resp.lower() for e in exp)

        by_category[cat]["total"] += 1
        if hit:
            by_category[cat]["hits"] += 1

        detailed_results.append({
            "id": pid,
            "category": cat,
            "expected": exp,
            "response": resp,
            "hit": hit
        })

    summary = {
        "total_probes": len(probes),
        "total_hits": sum(v["hits"] for v in by_category.values()),
        "accuracy": sum(v["hits"] for v in by_category.values()) / max(1, len(probes)),
        "by_category": {
            k: {
                "total": v["total"],
                "hits": v["hits"],
                "acc": v["hits"] / max(1, v["total"])
            }
            for k, v in by_category.items()
        }
    }
    return summary, detailed_results

def main():
    with open(BENCHMARK_DIR / "CR19_CONTEXT_200.json", "r", encoding="utf-8") as f:
        context_probes = json.load(f)
    with open(BENCHMARK_DIR / "CR19_CONVERSATION_100.json", "r", encoding="utf-8") as f:
        conv_probes = json.load(f)

    candidates = [
        {"name": "parent_baseline", "path": PARENT_CKPT, "arm": "baseline", "seed": 0},
        {"name": "arm_a_seed101", "path": CHECKPOINT_DIR / "arm_a" / "arm_a_seed101_best.pt", "arm": "arm_a", "seed": 101},
        {"name": "arm_a_seed202", "path": CHECKPOINT_DIR / "arm_a" / "arm_a_seed202_best.pt", "arm": "arm_a", "seed": 202},
        {"name": "arm_a_seed303", "path": CHECKPOINT_DIR / "arm_a" / "arm_a_seed303_best.pt", "arm": "arm_a", "seed": 303},
        {"name": "arm_b_seed101", "path": CHECKPOINT_DIR / "arm_b" / "arm_b_seed101_best.pt", "arm": "arm_b", "seed": 101},
        {"name": "arm_b_seed202", "path": CHECKPOINT_DIR / "arm_b" / "arm_b_seed202_best.pt", "arm": "arm_b", "seed": 202},
        {"name": "arm_b_seed303", "path": CHECKPOINT_DIR / "arm_b" / "arm_b_seed303_best.pt", "arm": "arm_b", "seed": 303},
    ]

    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    all_eval_summaries = {}
    all_raw_details = {}

    for cand in candidates:
        name = cand["name"]
        ckpt_path = cand["path"]
        print(f"\n=======================================================")
        print(f"  EVALUATING CANDIDATE: {name}")
        print(f"=======================================================")
        t0 = time.time()
        runner = DirectNeuralRunner(ckpt_path)
        cli = ConversationalChatCLI(ckpt_path)

        ctx_summary, ctx_details = evaluate_model_on_context(runner, cli, context_probes, cfg)
        print(f"Context 200 (CLI):    {ctx_summary['accuracy_cli']*100:.2f}% ({ctx_summary['total_hits_cli']}/{ctx_summary['total_probes']})")
        print(f"Context 200 (Direct): {ctx_summary['accuracy_direct']*100:.2f}% ({ctx_summary['total_hits_direct']}/{ctx_summary['total_probes']})")
        for g, d in sorted(ctx_summary["by_group"].items()):
            print(f"  {g:8s} | Direct: {d['hits_direct']:2d}/{d['total']} ({d['acc_direct']*100:.1f}%) | CLI: {d['hits_cli']:2d}/{d['total']} ({d['acc_cli']*100:.1f}%)")

        conv_summary, conv_details = evaluate_model_on_conversation(runner, conv_probes, cfg)
        print(f"Conversation 100:     {conv_summary['accuracy']*100:.2f}% ({conv_summary['total_hits']}/{conv_summary['total_probes']})")
        for cat, d in sorted(conv_summary["by_category"].items()):
            print(f"  {cat:15s} | {d['hits']:2d}/{d['total']} ({d['acc']*100:.1f}%)")

        elapsed = time.time() - t0
        print(f"Evaluation took {elapsed:.1f}s")

        all_eval_summaries[name] = {
            "candidate": name,
            "arm": cand["arm"],
            "seed": cand["seed"],
            "checkpoint_path": str(cand["path"].relative_to(REPO_ROOT)),
            "elapsed_sec": round(elapsed, 2),
            "context_summary": ctx_summary,
            "conversation_summary": conv_summary
        }
        all_raw_details[name] = {
            "context_details": ctx_details,
            "conversation_details": conv_details
        }

    with open(RESULTS_DIR / "all_eval_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_eval_summaries, f, indent=2)

    with open(RESULTS_DIR / "raw_generations_all.json", "w", encoding="utf-8") as f:
        json.dump(all_raw_details, f, indent=2)

    print("\n[SUCCESS] Successfully saved results to all_eval_summaries.json and raw_generations_all.json")

if __name__ == "__main__":
    main()
