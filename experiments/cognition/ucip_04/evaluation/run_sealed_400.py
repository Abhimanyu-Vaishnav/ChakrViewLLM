"""
Comparative Sealed Evaluation Harness for UCIP-04 across 400 probes.
Evaluates:
1. v0.11 Release Model
2. UCIP-03 Micro Release
3. UCIP-04 Micro Candidate
4. UCIP-04 Medium Candidate
Uses independent semantic evaluation rules (no length proxies).
"""

import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.model import ModelConfig, ChakrMicro
from scripts.run_m3_evaluations import load_tokenizer_artifacts, Sampler
from chakrview.runtime.sampling import SamplingConfig
from experiments.cognition.ucip_03r.independent_evaluation.independent_evaluator import evaluate_response

def run_sealed_benchmark():
    probes_file = REPO_ROOT / "experiments/cognition/ucip_04/evaluation/sealed_benchmark_400.json"
    with open(probes_file, "r", encoding="utf-8") as f:
        probes = json.load(f)

    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    sampler = Sampler()

    candidates = {
        "v0.11": REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt",
        "UCIP03_Micro": REPO_ROOT / "releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt",
        "UCIP04_Micro": REPO_ROOT / "experiments/cognition/ucip_04/checkpoints/micro/candidate_best.pt",
        "UCIP04_Medium": REPO_ROOT / "experiments/cognition/ucip_04/checkpoints/medium/candidate_best.pt"
    }

    all_summaries = {}
    all_raw = {}

    for c_name, ckpt_path in candidates.items():
        print(f"\n================ Evaluating: {c_name} on Sealed 400 ================")
        t0 = time.time()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_data = ckpt.get("config")
        if isinstance(cfg_data, dict):
            cfg = ModelConfig(**{k: v for k, v in cfg_data.items() if hasattr(ModelConfig, k)})
        else:
            cfg = cfg_data or ModelConfig()

        model = ChakrMicro(cfg)
        sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        suite_stats = {}
        probe_results = []

        for p in probes:
            suite = p["suite"]
            if suite not in suite_stats:
                suite_stats[suite] = {"total": 0, "pass": 0, "partial": 0, "fail": 0, "score_sum": 0.0}
            suite_stats[suite]["total"] += 1

            prompt_text = p["prompt"]
            pts = tok.encode(prompt_text, add_bos=True, add_eos=False)
            curr = list(pts)
            gt = []
            with torch.no_grad():
                for s in range(24):
                    tin = torch.tensor([curr], dtype=torch.long)
                    logits = model(tin)[0, -1, :]
                    nt = sampler.sample(logits, generated_tokens=gt, config=SamplingConfig(temperature=0.0, top_k=0), step=s)
                    gt.append(nt)
                    curr.append(nt)
                    if nt == 1:
                        break
            gen_text = tok.decode(gt, errors="replace").strip()

            cat_map = {
                "exact_copying": "copying",
                "simple_generation": "english",
                "basic_explanation": "english",
                "conversation_persona": "english",
                "devanagari_hindi": "devanagari_hindi",
                "roman_hindi": "roman_hindi",
                "hinglish": "hinglish",
                "multi_turn_context": "entity_recall",
                "repetition_resistance": "english"
            }
            cat = cat_map.get(suite, "english")

            res = evaluate_response(cat, prompt_text, gen_text, target=p["target"])
            v = res["verdict"]
            sc = res["score"]

            if v == "PASS":
                suite_stats[suite]["pass"] += 1
            elif v == "PARTIAL":
                suite_stats[suite]["partial"] += 1
            else:
                suite_stats[suite]["fail"] += 1
            suite_stats[suite]["score_sum"] += sc

            probe_results.append({
                "probe_id": p["id"],
                "suite": suite,
                "prompt": prompt_text,
                "target": p["target"],
                "generated": gen_text,
                "verdict": v,
                "score": sc,
                "reasoning": res["reasoning"]
            })

        dt = time.time() - t0
        total_p = len(probes)
        total_score = sum(s["score_sum"] for s in suite_stats.values())
        overall_pct = (total_score / total_p) * 100.0

        print(f"[{c_name}] Completed in {dt:.2f}s | Overall Sealed Score: {overall_pct:.2f}% ({total_score}/{total_p})")
        for s_name, st in suite_stats.items():
            pct = st["score_sum"] / st["total"] * 100.0
            print(f"  {s_name}: Pass={st['pass']}, Partial={st['partial']}, Fail={st['fail']} | Score={pct:.1f}%")

        all_summaries[c_name] = {
            "overall_score": overall_pct,
            "total_score": total_score,
            "total_probes": total_p,
            "suite_stats": suite_stats
        }
        all_raw[c_name] = probe_results

    out_sum = REPO_ROOT / "experiments/cognition/ucip_04/results/sealed_400_summary.json"
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    out_raw = REPO_ROOT / "experiments/cognition/ucip_04/results/sealed_400_raw.json"
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, indent=2, ensure_ascii=False)

    print(f"\nSaved sealed 400 summary to {out_sum}")
    print(f"Saved sealed 400 raw logs to {out_raw}")

if __name__ == "__main__":
    run_sealed_benchmark()
