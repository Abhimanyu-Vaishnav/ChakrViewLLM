"""
Batch 1781-1820: Work Package 1 & 2 Audit & Diagnostic Harness.
Analyzes every error in the 300-task benchmark and investigates the 10 adversarial prompt injections.
"""
import json
import os
from collections import Counter

BENCHMARK_RESULTS_PATH = "experiments/batch1741_1780/results/BENCHMARK_300_EVALUATION_RESULTS.json"
BENCHMARK_TASKS_PATH = "experiments/batch1741_1780/data/test_held_out_300.json"
TRAIN_TASKS_PATH = "experiments/batch1741_1780/data/train.jsonl"

def run_error_analysis():
    with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    with open(BENCHMARK_TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_map = {t["id"]: t for t in tasks}

    tasks_seed_42 = results["per_task_results"]["variant_b_seed_42"]
    
    analysis_records = []
    category_counts = Counter()
    adversarial_records = []
    
    for item in tasks_seed_42:
        tid = item["id"]
        task_info = task_map[tid]
        prompt = item["prompt"]
        true_intent = item["true_intent"]
        pred_intent = item["pred_intent"]
        family = item["family"]
        conf = item["confidence"]
        
        is_error = not item["match_intent"]
        
        # Categorize
        if "injection" in tid or "adversarial" in family:
            task_type = "adversarial_injection"
        elif "hard_negative" in family or "hard_negative" in tid:
            task_type = "hard_negative"
        elif true_intent == "CLARIFY":
            task_type = "incomplete_request"
        else:
            task_type = "ordinary_request"
            
        failure_type = "none"
        if is_error:
            if task_type == "adversarial_injection":
                failure_type = "adversarial_shortcut"
            elif true_intent == "CLARIFY":
                failure_type = "missing_argument_blindness"
            elif true_intent == "OUT_OF_SCOPE" and pred_intent in ["CODE_TASK", "CALCULATE", "VERIFY_CLAIM"]:
                failure_type = "lexical_distractor"
            elif true_intent == "CALCULATE" and pred_intent == "KNOWLEDGE_QUERY":
                failure_type = "conversational_math_confusion"
            elif true_intent == "VERIFY_CLAIM" and pred_intent in ["CALCULATE", "KNOWLEDGE_QUERY"]:
                failure_type = "factual_semantic_overlap"
            elif true_intent == "KNOWLEDGE_QUERY" and pred_intent in ["OUT_OF_SCOPE", "VERIFY_CLAIM"]:
                failure_type = "query_ambiguity"
            else:
                failure_type = "semantic_generalization_gap"
            category_counts[failure_type] += 1
            
        rec = {
            "id": tid,
            "family": family,
            "task_type": task_type,
            "prompt": prompt,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "match_intent": item["match_intent"],
            "arg_match": item["arg_match"],
            "confidence": conf,
            "failure_type": failure_type
        }
        analysis_records.append(rec)
        if task_type == "adversarial_injection":
            adversarial_records.append(rec)
            
    # Check training exposure to injections
    train_prompts = []
    with open(TRAIN_TASKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.append(json.loads(line))
                
    injection_tokens_in_train = []
    for tp in train_prompts:
        text = tp.get("input_text", tp.get("prompt", ""))
        p_lower = text.lower()
        if any(tok in p_lower for tok in ["override", "system", "ignore", "bypass", "jailbreak", "os.system", "shutil"]):
            injection_tokens_in_train.append(tp)
            
    output = {
        "total_tasks": len(analysis_records),
        "total_errors": sum(1 for r in analysis_records if not r["match_intent"]),
        "error_distribution": dict(category_counts),
        "adversarial_audit": adversarial_records,
        "train_exposure_count": len(injection_tokens_in_train),
        "train_exposure_samples": injection_tokens_in_train
    }
    
    with open("experiments/batch1781_1820/results/ERROR_ANALYSIS_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        
    print(f"Error analysis completed. Total errors: {output['total_errors']}/300")
    print(f"Error breakdown: {output['error_distribution']}")
    print(f"Adversarial count: {len(adversarial_records)}, all routed to: {Counter(r['pred_intent'] for r in adversarial_records)}")

if __name__ == "__main__":
    run_error_analysis()
