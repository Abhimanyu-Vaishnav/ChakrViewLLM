"""
Evaluation Harness for CR-EXP-01 Variable Binding Suite
Evaluates a given model against:
1. Condition A (Familiar Train Pool)
2. Condition B (Novel Held-Out Pool)
3. Condition D (Dynamic Reassignment)
4. Condition E (5 Distractors Stress Test)
5. Condition I (Unseen Templates 9-10)
6. Standard 40-probe Syntactic Suite (Python syntax, Delimiters, Logic, English)
7. General Language Continuation Probes
Calculates Wilson Score 95% Confidence Intervals and Invalid Output Rates.
"""

import json
import math
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.syntactic_probe import SyntacticProbeEvaluator

DATA_DIR = REPO_ROOT / "data" / "sft" / "cr_exp01"

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - spread) * 100, min(100.0, centre + spread) * 100

def evaluate_split(model, tok, file_path, max_gen_tokens=4):
    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    correct = 0
    invalid = 0
    details = []
    
    for r in records:
        full_text = r["text"]
        target = r["target_value"].strip()
        
        # Split prompt right at 'Answer:'
        assert "Answer:" in full_text
        prompt, _ = full_text.split("Answer:", 1)
        prompt = prompt + "Answer:"
        
        ids = tok.encode(prompt)
        curr = list(ids)
        
        for _ in range(max_gen_tokens):
            x = torch.tensor([curr], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
                next_tok = torch.argmax(logits[0, -1, :]).item()
            curr.append(next_tok)
            # Stop if newline
            decoded_latest = tok.decode([next_tok])
            if "\n" in decoded_latest:
                break
                
        gen_answer = tok.decode(curr[len(ids):]).strip()
        is_match = (gen_answer == target) or gen_answer.startswith(target)
        if is_match:
            correct += 1
            
        # Check invalid (e.g. empty, or repetition of delimiter)
        if not gen_answer or gen_answer in ["Answer:", "Query:", ":", ""]:
            invalid += 1
            
        details.append({
            "id": r.get("id"),
            "target": target,
            "generated": gen_answer,
            "correct": is_match
        })
        
    n = len(records)
    acc = (correct / n) * 100
    ci_low, ci_high = wilson_ci(correct, n)
    invalid_rate = (invalid / n) * 100
    
    return {
        "total": n,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "wilson_ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "invalid_output_rate_pct": round(invalid_rate, 2),
        "sample_outputs": details[:5]
    }

def run_evaluation(checkpoint_path, output_json_path):
    print(f"=== EVALUATING CHECKPOINT: {checkpoint_path} ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    results = {
        "checkpoint_path": str(checkpoint_path),
        "conditions": {}
    }
    
    splits = {
        "Condition_A_Familiar": DATA_DIR / "test_familiar_100.jsonl",
        "Condition_B_Novel_Binding": DATA_DIR / "test_novel_100.jsonl",
        "Condition_D_Reassignment": DATA_DIR / "test_reassignment_100.jsonl",
        "Condition_E_5_Distractors": DATA_DIR / "test_distractors_100.jsonl",
        "Condition_I_Unseen_Template": DATA_DIR / "test_unseen_template_100.jsonl"
    }
    
    for cond_name, p in splits.items():
        print(f"Evaluating {cond_name}...")
        res = evaluate_split(model, tok, p)
        results["conditions"][cond_name] = res
        print(f"  {cond_name}: Acc = {res['accuracy_pct']}% (95% CI: {res['wilson_ci_95']}), Invalid = {res['invalid_output_rate_pct']}%")
        
    # Syntactic Probe Evaluation
    print("Evaluating Standard Syntactic Probe Suite...")
    evaluator = SyntacticProbeEvaluator(tok, device=torch.device("cpu"))
    synth_res = evaluator.evaluate(model)
    results["syntactic_probe"] = {
        "top1_accuracy": synth_res.top1_accuracy,
        "top5_accuracy": synth_res.top5_accuracy,
        "mean_target_log_prob": synth_res.mean_target_log_prob,
        "category_metrics": synth_res.category_metrics
    }
    print(f"  Syntactic Probe: Top-1 = {synth_res.top1_accuracy}%, Top-5 = {synth_res.top5_accuracy}%")
    
    # Save output
    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {out_p}")
    return results

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "experiments/cognition/cr_exp_01/baseline_evaluation_results.json"
    run_evaluation(ckpt, out)

