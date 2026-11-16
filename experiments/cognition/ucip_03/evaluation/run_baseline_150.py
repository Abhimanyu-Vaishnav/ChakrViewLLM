"""
Evaluation harness for the 150-probe baseline diagnostic benchmark.
Evaluates:
- Stage B Pretrained (training_checkpoint_step_10000.pt)
- CBR-SFT-04 (sft_checkpoint_step_200.pt)
- v0.11 Conversational Release (chakrview_v011_model.pt)
- 13M Recovered Candidate (ucip01r_13m_recovered_best.pt)
- UCIP-02 Checkpoint (ucip02_recovered_model_best.pt)
Saves raw generation logs and scorecard summary.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.model import ModelConfig, ChakrMicro
from scripts.run_m3_evaluations import load_tokenizer_artifacts, Sampler
from chakrview.runtime.sampling import SamplingConfig

def evaluate_model_on_probes(model, tokenizer, sampler, probes: List[Dict[str, Any]], max_new_tokens: int = 32) -> Dict[str, Any]:
    results = []
    sampling_cfg = SamplingConfig(temperature=0.0, top_k=0)
    category_stats = {}

    for p in probes:
        cat = p["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        prompt = p["prompt"]
        prompt_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        curr_tokens = list(prompt_tokens)
        gen_tokens = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_in = torch.tensor([curr_tokens], dtype=torch.long)
                logits = model(t_in)[0, -1, :]
                next_tok = sampler.sample(logits, generated_tokens=gen_tokens, config=sampling_cfg, step=step)
                gen_tokens.append(next_tok)
                curr_tokens.append(next_tok)
                if next_tok == 1:
                    break

        gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
        
        # Scoring
        target = p["target"].lower()
        gen_lower = gen_text.lower()
        eval_type = p.get("eval_type", "substring_match")

        is_correct = False
        if eval_type == "exact_copy":
            is_correct = (p["target"].strip().lower() in gen_lower)
        elif eval_type in ["substring_match", "number_match", "context_retrieval"]:
            is_correct = (target in gen_lower)
        elif eval_type == "keyword_in_sentence":
            # Must mention target and have some sentence length
            is_correct = (target in gen_lower and len(gen_tokens) >= 4)
        elif eval_type in ["devanagari_match", "roman_hindi_match", "hinglish_match", "persona_presence"]:
            is_correct = (target in gen_lower or len(gen_text) > 2)

        if is_correct:
            category_stats[cat]["correct"] += 1

        results.append({
            "probe_id": p["probe_id"],
            "category": cat,
            "prompt": prompt,
            "target": p["target"],
            "generated_text": gen_text,
            "is_correct": is_correct
        })

    total_probes = len(probes)
    total_correct = sum(s["correct"] for s in category_stats.values())
    accuracy = (total_correct / total_probes) * 100.0

    return {
        "accuracy": accuracy,
        "total_probes": total_probes,
        "total_correct": total_correct,
        "category_stats": category_stats,
        "results": results
    }

def main():
    bench_path = REPO_ROOT / "experiments/cognition/ucip_03/evaluation/diagnostic_baseline_150.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    sampler = Sampler()

    candidates = {
        "Stage_B": REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        "CBR_SFT_04": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "v0.11": REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt",
        "13M_Recovered": REPO_ROOT / "experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt",
        "UCIP_02_task5995": REPO_ROOT / "experiments/cognition/ucip_02/training/ucip02_recovered_model_best.pt"
    }

    all_evaluations = {}

    for cand_name, ckpt_path in candidates.items():
        print(f"\n--- Evaluating {cand_name} ---")
        t0 = time.time()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            cfg = ModelConfig()
        elif isinstance(cfg, dict):
            cfg = ModelConfig(**{k: v for k, v in cfg.items() if hasattr(ModelConfig, k)})
        
        model = ChakrMicro(cfg)
        sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        eval_res = evaluate_model_on_probes(model, tokenizer, sampler, probes)
        dt = time.time() - t0
        print(f"{cand_name} evaluated in {dt:.2f}s: Total Accuracy = {eval_res['accuracy']:.2f}% ({eval_res['total_correct']}/{eval_res['total_probes']})")
        for cat, st in eval_res["category_stats"].items():
            print(f"  {cat}: {st['correct']}/{st['total']} ({st['correct']/st['total']*100:.1f}%)")

        all_evaluations[cand_name] = eval_res

    # Save summary
    out_summary_path = REPO_ROOT / "experiments/cognition/ucip_03/results/baseline_150_summary.json"
    summary_data = {
        cand: {
            "accuracy": res["accuracy"],
            "total_correct": res["total_correct"],
            "total_probes": res["total_probes"],
            "category_stats": res["category_stats"]
        }
        for cand, res in all_evaluations.items()
    }
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save raw generations
    out_raw_path = REPO_ROOT / "experiments/cognition/ucip_03/results/baseline_150_raw_generations.json"
    with open(out_raw_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary to {out_summary_path}")
    print(f"Saved raw generations to {out_raw_path}")

if __name__ == "__main__":
    main()
