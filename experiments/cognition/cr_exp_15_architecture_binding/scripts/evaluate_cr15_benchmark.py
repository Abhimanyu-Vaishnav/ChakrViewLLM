"""CR-EXP-15 Benchmark Evaluation Suite
Evaluates 7 models on the 1,200 frozen benchmark probes across 8 decomposed capability tasks.
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path("D:/Project/ChakrView")
EXP_DIR = REPO_ROOT / "experiments/cognition/cr_exp_15_architecture_binding"
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
BENCHMARK_PATH = EXP_DIR / "benchmark_data/benchmark_1200_sealed.json"
EVAL_OUT_DIR = EXP_DIR / "eval_outputs"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "baseline_parent": {
        "ckpt_path": REPO_ROOT / "experiments/cognition/cr_exp_14_entity_binding/checkpoints/baseline_cr12_b/cr_exp_12_candidate_b_best_val_copy.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512),
        "desc": "CR-12 Candidate B Parent (192-dim, 6 heads, d_head=32)"
    },
    "candidate_a": {
        "ckpt_path": EXP_DIR / "checkpoints/candidate_a/candidate_a_best_val.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512),
        "desc": "Candidate A: Standard LM Control (Parent + General LM)"
    },
    "candidate_b": {
        "ckpt_path": EXP_DIR / "checkpoints/candidate_b/candidate_b_best_val.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512),
        "desc": "Candidate B: Copy & Select Curriculum"
    },
    "candidate_c": {
        "ckpt_path": EXP_DIR / "checkpoints/candidate_c/candidate_c_best_val.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512),
        "desc": "Candidate C: Static Binding Curriculum"
    },
    "candidate_d": {
        "ckpt_path": EXP_DIR / "checkpoints/candidate_d/candidate_d_best_val.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512),
        "desc": "Candidate D / Arch Arm A: Dynamic Updates Curriculum (192-dim, 6 heads, d_head=32)"
    },
    "arch_arm_b_heads3_d64": {
        "ckpt_path": EXP_DIR / "checkpoints/arch_arm_b_heads3_d64/arch_arm_b_heads3_d64_best_val.pt",
        "config": ModelConfig(d_model=192, n_layers=6, n_heads=3, hidden_dim=512),
        "desc": "Arch Arm B: 192-dim, 3 heads, d_head=64"
    },
    "arch_arm_c_width256_d64": {
        "ckpt_path": EXP_DIR / "checkpoints/arch_arm_c_width256_d64/arch_arm_c_width256_d64_best_val.pt",
        "config": ModelConfig(d_model=256, n_layers=6, n_heads=4, hidden_dim=682),
        "desc": "Arch Arm C: 256-dim, 4 heads, d_head=64 (5.77M params)"
    }
}

def load_benchmark():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 40, device: torch.device = torch.device("cpu")) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > 250:
                input_ids = input_ids[:, -250:]
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            
            # Stop on 0 (PAD) or newline after output
            if next_token == 0:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            text = tokenizer.decode(generated, errors="replace")
            # Stop condition if generation contains newline or next speaker
            if "\nUser:" in text or "\nTurn" in text or "\n" in text:
                break
                
    return tokenizer.decode(generated, errors="replace").strip()

def evaluate_probe(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    category = probe.get("category", "")
    target_entity = probe.get("subject", "").strip()
    expected_value = probe.get("expected_target", "").strip()
    is_abstention = (category == "calibrated_abstention" or expected_value == "abstain")
    
    out_lower = raw_output.lower()
    
    if is_abstention:
        # Check calibrated abstention
        abstention_cues = [
            "not mentioned", "unknown", "never mentioned", "no record", 
            "cannot be determined", "not specified", "unspecified", "i don't know",
            "not stated", "no information", "has not been provided", "not discussed"
        ]
        abstained = any(cue in out_lower for cue in abstention_cues)
        correct = abstained
        return {
            "correct": correct,
            "has_subject": True if abstained else False,
            "has_value": not abstained,
            "abstained": abstained,
            "raw_output": raw_output
        }
    
    target_lower = target_entity.lower()
    value_lower = expected_value.lower()
    
    has_subject = target_lower in out_lower if target_lower else True
    has_value = value_lower in out_lower if value_lower else False
    
    # Strict accuracy: requires both subject and value
    strict_correct = has_subject and has_value
    
    return {
        "correct": strict_correct,
        "has_subject": has_subject,
        "has_value": has_value,
        "abstained": False,
        "raw_output": raw_output
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    probes = load_benchmark()
    print(f"Loaded {len(probes)} benchmark probes.")
    
    summary_results = {}
    
    for model_key, spec in MODEL_SPECS.items():
        print(f"\n=======================================================")
        print(f"Evaluating: {model_key} ({spec['desc']})")
        print(f"Loading checkpoint from: {spec['ckpt_path']}")
        
        if not spec['ckpt_path'].exists():
            print(f"ERROR: Checkpoint {spec['ckpt_path']} does not exist!")
            continue
            
        model = ChakrMicro(spec['config'])
        state = torch.load(spec['ckpt_path'], map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        probe_evaluations = []
        task_stats = {}
        
        for idx, probe in enumerate(probes):
            prompt = probe["prompt"]
            if not prompt.endswith("Assistant: ") and not prompt.endswith("Assistant:"):
                prompt = prompt.rstrip() + "\nAssistant: "
            raw_out = generate_greedy(model, tokenizer, prompt, max_new_tokens=30, device=device)
            eval_res = evaluate_probe(probe, raw_out)
            eval_res["id"] = probe["id"]
            eval_res["category"] = probe["category"]
            eval_res["subject"] = probe.get("subject", "")
            eval_res["expected_target"] = probe.get("expected_target", "")
            eval_res["is_abstention"] = (probe["category"] == "calibrated_abstention" or probe.get("expected_target") == "abstain")
            probe_evaluations.append(eval_res)
            
            cat = probe["category"]
            if cat not in task_stats:
                task_stats[cat] = {
                    "total": 0,
                    "correct": 0,
                    "subject_correct": 0,
                    "value_correct": 0
                }
            task_stats[cat]["total"] += 1
            if eval_res["correct"]:
                task_stats[cat]["correct"] += 1
            if eval_res["has_subject"]:
                task_stats[cat]["subject_correct"] += 1
            if eval_res["has_value"]:
                task_stats[cat]["value_correct"] += 1
                
            if (idx + 1) % 300 == 0:
                print(f"Evaluated {idx + 1}/{len(probes)} probes...")
                
        total_probes = len(probes)
        total_correct = sum(1 for p in probe_evaluations if p["correct"])
        non_abstention_probes = [p for p in probe_evaluations if not p["is_abstention"]]
        abstention_probes = [p for p in probe_evaluations if p["is_abstention"]]
        
        non_abs_strict = sum(1 for p in non_abstention_probes if p["correct"])
        non_abs_subj = sum(1 for p in non_abstention_probes if p["has_subject"])
        non_abs_val = sum(1 for p in non_abstention_probes if p["has_value"])
        abs_correct = sum(1 for p in abstention_probes if p["correct"])
        
        model_summary = {
            "model_key": model_key,
            "description": spec["desc"],
            "total_probes": total_probes,
            "overall_accuracy": total_correct / total_probes,
            "non_abstention_total": len(non_abstention_probes),
            "non_abstention_strict_acc": non_abs_strict / len(non_abstention_probes),
            "non_abstention_subject_acc": non_abs_subj / len(non_abstention_probes),
            "non_abstention_value_acc": non_abs_val / len(non_abstention_probes),
            "abstention_total": len(abstention_probes),
            "abstention_acc": abs_correct / len(abstention_probes),
            "task_breakdown": {
                t: {
                    "total": s["total"],
                    "accuracy": s["correct"] / s["total"],
                    "subject_acc": s["subject_correct"] / s["total"],
                    "value_acc": s["value_correct"] / s["total"]
                }
                for t, s in task_stats.items()
            }
        }
        
        summary_results[model_key] = model_summary
        
        out_file = EVAL_OUT_DIR / f"{model_key}_eval.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"summary": model_summary, "probes": probe_evaluations}, f, indent=2)
            
        print(f"Results for {model_key}:")
        print(f"  Overall Acc: {model_summary['overall_accuracy']:.4f} ({total_correct}/{total_probes})")
        print(f"  Non-Abstention Strict Acc: {model_summary['non_abstention_strict_acc']:.4f} ({non_abs_strict}/{len(non_abstention_probes)})")
        print(f"  Non-Abstention Subject Acc: {model_summary['non_abstention_subject_acc']:.4f} ({non_abs_subj}/{len(non_abstention_probes)})")
        print(f"  Non-Abstention Value Acc: {model_summary['non_abstention_value_acc']:.4f} ({non_abs_val}/{len(non_abstention_probes)})")
        print(f"  Abstention Acc: {model_summary['abstention_acc']:.4f} ({abs_correct}/{len(abstention_probes)})")

    summary_path = EVAL_OUT_DIR / "all_models_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nAll evaluations completed! Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
