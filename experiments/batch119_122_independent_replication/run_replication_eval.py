"""
Independent replication evaluation runner comparing:
Arm 1: Canonical Base Model trained with Balanced Curriculum (balanced_control_base_step150.pt)
Arm 2: Copy-Task Pre-Trained Model trained with Balanced Curriculum (balanced_curriculum_bridge_step150.pt)

Evaluates cold-reloaded weights on independent_eval.jsonl using free-running greedy decoding.
Computes multi-dimensional metrics, breakdowns by subcategory, and full per-example logs.
"""
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    evaluate_single_prediction,
    compute_aggregate_metrics,
)

EVAL_DATA_PATH = Path("experiments/batch119_122_independent_replication/data/independent_eval.jsonl")
REPORTS_DIR = Path("experiments/batch119_122_independent_replication/reports")

# Checkpoint paths
ARM1_CKPT = Path("experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_control_base_step150.pt")
ARM2_CKPT = Path("experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_curriculum_bridge_step150.pt")

def load_eval_records():
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def evaluate_checkpoint(ckpt_path: Path, condition_name: str, records: list, tokenizer):
    print(f"\n{'='*70}")
    print(f"Evaluating {condition_name}")
    print(f"Path: {ckpt_path}")
    sha = compute_file_sha256(str(ckpt_path))
    print(f"SHA-256: {sha}")
    print(f"{'='*70}")
    
    # Cold reload
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    
    eval_list = []
    for r in records:
        prompt_ids = [0] + tokenizer.encode(r["prompt"])
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(50):
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
                    
        raw_gen = tokenizer.decode(gen_tokens[len(prompt_ids):])
        clean_gen = raw_gen.split("<|im_end|>")[0].strip()
        
        res = evaluate_single_prediction(clean_gen, r)
        res["id"] = r["id"]
        res["category"] = r.get("category", "")
        res["prompt_snippet"] = r["prompt"].replace("\n", " ")[:60]
        res["full_prediction"] = clean_gen
        eval_list.append(res)
        
    # Aggregate over entire evaluation set
    overall_agg = compute_aggregate_metrics(eval_list)
    overall_agg["condition_name"] = condition_name
    overall_agg["checkpoint_path"] = str(ckpt_path)
    overall_agg["checkpoint_sha256"] = sha
    
    # Compute subcategory breakdowns
    categories = sorted(list({r.get("category", "") for r in records}))
    cat_breakdown = {}
    for cat in categories:
        cat_records = [r for r in eval_list if r["category"] == cat]
        cat_agg = compute_aggregate_metrics(cat_records)
        cat_breakdown[cat] = cat_agg
        
    overall_agg["subcategory_breakdown"] = cat_breakdown
    overall_agg["per_example_details"] = eval_list
    
    return overall_agg

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_eval_records()
    print(f"Loaded {len(records)} independent evaluation records.")
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # Evaluate Arm 1: Canonical Base Control
    res_arm1 = evaluate_checkpoint(ARM1_CKPT, "canonical_base_curriculum_control", records, tokenizer)
    
    # Evaluate Arm 2: Copy-Initialized Curriculum Bridge
    res_arm2 = evaluate_checkpoint(ARM2_CKPT, "copy_initialized_curriculum_bridge", records, tokenizer)
    
    combined_report = {
        "benchmark": "batch121_independent_held_out_replication",
        "eval_dataset": str(EVAL_DATA_PATH),
        "total_test_records": len(records),
        "canonical_base_control": res_arm1,
        "copy_initialized_curriculum_bridge": res_arm2,
    }
    
    report_file = REPORTS_DIR / "replication_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
        
    print(f"\nReplication complete. Results saved to {report_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF INDEPENDENT REPLICATION RESULTS")
    print("="*80)
    print(f"{'Metric':<35} | {'Arm 1 (Canonical)':<20} | {'Arm 2 (Curriculum Bridge)':<25}")
    print("-" * 85)
    
    t1 = res_arm1.get("tool_metrics", {})
    t2 = res_arm2.get("tool_metrics", {})
    nt1 = res_arm1.get("non_tool_metrics", {})
    nt2 = res_arm2.get("non_tool_metrics", {})
    
    print(f"{'Tool Parse Rate':<35} | {t1.get('parse_rate', 0):<20} | {t2.get('parse_rate', 0):<25}")
    print(f"{'Tool Schema Validity':<35} | {t1.get('schema_validity_rate', 0):<20} | {t2.get('schema_validity_rate', 0):<25}")
    print(f"{'Tool Accuracy':<35} | {t1.get('tool_accuracy', 0):<20} | {t2.get('tool_accuracy', 0):<25}")
    print(f"{'Operation Accuracy':<35} | {t1.get('operation_accuracy', 0):<20} | {t2.get('operation_accuracy', 0):<25}")
    print(f"{'Operand A Accuracy':<35} | {t1.get('operand_a_accuracy', 0):<20} | {t2.get('operand_a_accuracy', 0):<25}")
    print(f"{'Operand B Accuracy':<35} | {t1.get('operand_b_accuracy', 0):<20} | {t2.get('operand_b_accuracy', 0):<25}")
    print(f"{'Exact-Call Accuracy':<35} | {t1.get('exact_call_accuracy', 0):<20} | {t2.get('exact_call_accuracy', 0):<25}")
    print(f"{'Execution Success Rate':<35} | {t1.get('execution_success_rate', 0):<20} | {t2.get('execution_success_rate', 0):<25}")
    print(f"{'End-to-End Correctness':<35} | {t1.get('e2e_correctness_rate', 0):<20} | {t2.get('e2e_correctness_rate', 0):<25}")
    print(f"{'Unique Outputs / Total':<35} | {t1.get('unique_outputs_count', 0)}/{t1.get('denominator', 0)} | {t2.get('unique_outputs_count', 0)}/{t2.get('denominator', 0)}")
    print(f"{'Collapse Rate':<35} | {t1.get('collapse_rate', 0):<20} | {t2.get('collapse_rate', 0):<25}")
    print("-" * 85)
    print(f"{'Non-Tool Total False Positive':<35} | {nt1.get('total_false_positive_rate', 0):<20} | {nt2.get('total_false_positive_rate', 0):<25}")
    print(f"{'Non-Tool Valid False Positive':<35} | {nt1.get('valid_false_positive_rate', 0):<20} | {nt2.get('valid_false_positive_rate', 0):<25}")
    print(f"{'Non-Tool Malformed False Pos':<35} | {nt1.get('malformed_false_positive_rate', 0):<20} | {nt2.get('malformed_false_positive_rate', 0):<25}")
    print(f"{'Non-Tool Clean Refusal Rate':<35} | {nt1.get('clean_refusal_rate', 0):<20} | {nt2.get('clean_refusal_rate', 0):<25}")
    print("=" * 80)

if __name__ == "__main__":
    main()
