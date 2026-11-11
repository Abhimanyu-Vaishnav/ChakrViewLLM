"""CR-EXP-15A Raw Prediction Independent Rescoring Script
Loads the exact frozen benchmark and raw predictions from CR-EXP-15,
and re-scores them using the calibrated evaluator across all decomposed capabilities.
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP15_DIR = REPO_ROOT / "experiments/cognition/cr_exp_15_architecture_binding"
AUDIT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_15a_forensic_audit"
sys.path.insert(0, str(AUDIT_DIR))

from calibrated_evaluator import CalibratedEvaluator

BENCHMARK_PATH = EXP15_DIR / "benchmark_data/benchmark_1200_sealed.json"

MODELS = [
    "baseline_parent",
    "candidate_a",
    "candidate_b",
    "candidate_c",
    "candidate_d",
    "arch_arm_b_heads3_d64",
    "arch_arm_c_width256_d64"
]

def main():
    evaluator = CalibratedEvaluator()
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    print(f"Loaded {len(benchmark)} frozen benchmark probes.")
    
    full_rescore_report = {}
    
    for model_key in MODELS:
        eval_path = EXP15_DIR / f"eval_outputs/{model_key}_eval.json"
        if not eval_path.exists():
            print(f"Warning: {eval_path} not found.")
            continue
            
        with open(eval_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        probes_eval = raw_data["probes"]
        
        # Decomposed tracking counters
        stats = {
            "total": len(probes_eval),
            "literal_copying": {"total": 0, "correct": 0},
            "single_token_copying": {"total": 0, "correct": 0},
            "multi_token_copying": {"total": 0, "correct": 0},
            "entity_selection": {"total": 0, "correct": 0},
            "static_binding": {"total": 0, "correct": 0, "subject_correct": 0, "value_correct": 0},
            "single_entity_updates": {"total": 0, "correct": 0, "subject_correct": 0, "value_correct": 0, "stale_retained": 0},
            "multi_entity_tracking": {"total": 0, "correct": 0, "subject_correct": 0, "value_correct": 0, "stale_retained": 0},
            "interleaved_updates": {"total": 0, "correct": 0, "subject_correct": 0, "value_correct": 0, "stale_retained": 0},
            "delayed_retrieval_distractors": {"total": 0, "correct": 0, "subject_correct": 0, "value_correct": 0, "distractor_retained": 0},
            "answerable_probes": {"total": 0, "strict_correct": 0, "loose_correct": 0, "subject_correct": 0, "value_correct": 0},
            "calibrated_abstention": {"total": 0, "abstained": 0, "correct": 0},
            "false_abstentions": {"total": 0, "count": 0},
            "empty_generations": {"total": 0, "count": 0}
        }
        
        # In tokenizer vocab_4096:
        # Dev, Roy, Prem, Amit, Alok, Sonu, Mona, Ritu are single tokens.
        single_token_names = {"Dev", "Roy", "Prem", "Amit", "Alok", "Sonu", "Mona", "Ritu"}
        
        rescored_probes = []
        
        for p, b in zip(probes_eval, benchmark):
            task = b["category"]
            prompt = b["prompt"]
            raw_out = p["raw_output"]
            subj = b.get("subject", "")
            tgt = b.get("expected_target", "")
            earlier = b.get("earlier_value", "")
            distractors = b.get("distractor_values", [])
            is_abs = (task == "calibrated_abstention" or tgt == "abstain")
            
            res = evaluator.evaluate(
                task_type=task,
                prompt=prompt,
                raw_output=raw_out,
                expected_subject=subj,
                expected_attribute="location",
                expected_value=tgt,
                earlier_value=earlier,
                distractor_values=distractors,
                is_abstention=is_abs
            )
            res["id"] = b["id"]
            rescored_probes.append(res)
            
            # Aggregate stats
            if not raw_out.strip():
                stats["empty_generations"]["count"] += 1
                
            if is_abs:
                stats["calibrated_abstention"]["total"] += 1
                if res["abstained"]:
                    stats["calibrated_abstention"]["abstained"] += 1
                if res["abstention_correct"]:
                    stats["calibrated_abstention"]["correct"] += 1
            else:
                stats["answerable_probes"]["total"] += 1
                if res["strict_correct"]:
                    stats["answerable_probes"]["strict_correct"] += 1
                if res["loose_correct"]:
                    stats["answerable_probes"]["loose_correct"] += 1
                if res["subject_correct"]:
                    stats["answerable_probes"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["answerable_probes"]["value_correct"] += 1
                if res["abstained"]:
                    stats["false_abstentions"]["count"] += 1
                    
            if task == "entity_copying":
                stats["literal_copying"]["total"] += 1
                if res["strict_correct"]:
                    stats["literal_copying"]["correct"] += 1
                if subj in single_token_names:
                    stats["single_token_copying"]["total"] += 1
                    if res["strict_correct"]:
                        stats["single_token_copying"]["correct"] += 1
                else:
                    stats["multi_token_copying"]["total"] += 1
                    if res["strict_correct"]:
                        stats["multi_token_copying"]["correct"] += 1
            elif task == "in_context_selection":
                stats["entity_selection"]["total"] += 1
                if res["strict_correct"]:
                    stats["entity_selection"]["correct"] += 1
            elif task == "static_binding":
                stats["static_binding"]["total"] += 1
                if res["strict_correct"]:
                    stats["static_binding"]["correct"] += 1
                if res["subject_correct"]:
                    stats["static_binding"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["static_binding"]["value_correct"] += 1
            elif task == "single_entity_updates":
                stats["single_entity_updates"]["total"] += 1
                if res["strict_correct"]:
                    stats["single_entity_updates"]["correct"] += 1
                if res["subject_correct"]:
                    stats["single_entity_updates"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["single_entity_updates"]["value_correct"] += 1
                if res["stale_value_retained"]:
                    stats["single_entity_updates"]["stale_retained"] += 1
            elif task == "multi_entity_tracking":
                stats["multi_entity_tracking"]["total"] += 1
                if res["strict_correct"]:
                    stats["multi_entity_tracking"]["correct"] += 1
                if res["subject_correct"]:
                    stats["multi_entity_tracking"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["multi_entity_tracking"]["value_correct"] += 1
                if res["stale_value_retained"]:
                    stats["multi_entity_tracking"]["stale_retained"] += 1
            elif task == "interleaved_updates":
                stats["interleaved_updates"]["total"] += 1
                if res["strict_correct"]:
                    stats["interleaved_updates"]["correct"] += 1
                if res["subject_correct"]:
                    stats["interleaved_updates"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["interleaved_updates"]["value_correct"] += 1
                if res["stale_value_retained"]:
                    stats["interleaved_updates"]["stale_retained"] += 1
            elif task == "delayed_retrieval_distractors":
                stats["delayed_retrieval_distractors"]["total"] += 1
                if res["strict_correct"]:
                    stats["delayed_retrieval_distractors"]["correct"] += 1
                if res["subject_correct"]:
                    stats["delayed_retrieval_distractors"]["subject_correct"] += 1
                if res["value_correct"]:
                    stats["delayed_retrieval_distractors"]["value_correct"] += 1
                if res["distractor_retained"]:
                    stats["delayed_retrieval_distractors"]["distractor_retained"] += 1

        full_rescore_report[model_key] = stats
        
        # Save individual model rescore
        with open(AUDIT_DIR / f"{model_key}_rescored.json", "w", encoding="utf-8") as f:
            json.dump({"summary": stats, "probes": rescored_probes}, f, indent=2)

    with open(AUDIT_DIR / "rescoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_rescore_report, f, indent=2)
        
    print("\nRescoring completed successfully across all models!")
    for m, s in full_rescore_report.items():
        ans_tot = s["answerable_probes"]["total"]
        ans_strict = s["answerable_probes"]["strict_correct"]
        ans_loose = s["answerable_probes"]["loose_correct"]
        abs_tot = s["calibrated_abstention"]["total"]
        abs_cor = s["calibrated_abstention"]["correct"]
        print(f"{m}:")
        print(f"  Answerable Strict Acc: {ans_strict}/{ans_tot} ({ans_strict/ans_tot*100:.2f}%)")
        print(f"  Answerable Loose Acc:  {ans_loose}/{ans_tot} ({ans_loose/ans_tot*100:.2f}%)")
        print(f"  Abstention Acc:        {abs_cor}/{abs_tot} ({abs_cor/abs_tot*100:.2f}%)")

if __name__ == "__main__":
    main()

