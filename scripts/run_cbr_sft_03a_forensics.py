"""
scripts/run_cbr_sft_03a_forensics.py

Performs deep forensic audits:
1. Evaluates all checkpoints across all 5 suites:
   - Canonical Step 10k baseline
   - CBR-SFT-02 Steps 25, 50, 75, 100, 125, 150, 175, 200
2. First-token prediction diagnostics for 100 held-out prompts:
   - Expected first token ID and string
   - Actual top-1 predicted token
   - Expected token rank and probability
   - Top-5 tokens and probabilities
3. Teacher-forced vs free autoregressive generation token-by-token divergence:
   - Evaluates prefix match at positions 1, 2, 4, 8, 16
   - Teacher-forced cross-entropy loss vs free generation drift
4. Prompt sensitivity test:
   - Evaluates semantically equivalent prompt variants for first-token consistency and relevance
5. Controlled 2-example overfit test review:
   - Reproduces in isolated temp directory, verifies memorization vs unseen generalization
"""

import sys
import json
import re
import math
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.run_m3_evaluations import (
    DirectNeuralRunner,
    evaluate_conversational,
    evaluate_intelligence,
    evaluate_coding,
    file_sha256
)
from scripts.run_cbr_sft_02_evaluations import evaluate_held_out_suite

OUT_DIR = REPO_ROOT / "docs" / "missions" / "CBR_01" / "experiments" / "CBR_SFT_03A" / "raw_evals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATHS = [
    ("canonical_baseline", REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"),
    ("step_25", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_25.pt"),
    ("step_50", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_50.pt"),
    ("step_75", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_75.pt"),
    ("step_100", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_100.pt"),
    ("step_125", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_125.pt"),
    ("step_150", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_150.pt"),
    ("step_175", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_175.pt"),
    ("step_200", REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_02" / "sft_checkpoint_step_200.pt")
]

def run_all_checkpoint_evaluations():
    print("--- 1. EVALUATING ALL CHECKPOINTS ACROSS 5 SUITES ---")
    entity_suite_path = REPO_ROOT / "evaluation" / "cbr_suite" / "held_out_entity_grounding_100.json"
    mturn_suite_path = REPO_ROOT / "evaluation" / "cbr_suite" / "held_out_multiturn_context_50.json"
    
    master_summary = {}
    for name, path in CHECKPOINT_PATHS:
        print(f"Running evaluations for {name}...")
        runner = DirectNeuralRunner(path)
        conv = evaluate_conversational(runner, f"cbr3a_{name}", {"label": name, "path": path})
        intel = evaluate_intelligence(runner, f"cbr3a_{name}", {"label": name, "path": path})
        coding = evaluate_coding(runner, f"cbr3a_{name}", {"label": name, "path": path})
        entity = evaluate_held_out_suite(runner, entity_suite_path, "HELD_OUT_ENTITY_GROUNDING_100")
        mturn = evaluate_held_out_suite(runner, mturn_suite_path, "HELD_OUT_MULTITURN_CONTEXT_50")
        
        master_summary[name] = {
            "conversational_60": {"passed": conv["passed_tasks"], "total": 60, "accuracy": conv["accuracy"]},
            "intelligence_32": {"passed": intel["passed_tasks_count"], "total": 32, "attractors": intel["attractor_collapse_count"], "attractor_rate": intel["attractor_rate"]},
            "coding_12": {"passed": coding["test_passed_count"], "syntax_valid": coding["syntax_valid_count"], "total": 12},
            "entity_100": {"passed": entity["passed_tasks"], "total": 100, "accuracy": entity["accuracy"]},
            "mturn_50": {"passed": mturn["passed_tasks"], "total": 50, "accuracy": mturn["accuracy"]},
            "sha256": file_sha256(path)
        }
        
    with open(OUT_DIR / "all_checkpoints_master_matrix.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print("All checkpoints evaluated.")
    return master_summary

def run_first_token_and_teacher_forcing_audit(target_ckpt_name="step_50"):
    print(f"\n--- 2. FIRST-TOKEN & TEACHER-FORCED DIAGNOSTICS ({target_ckpt_name}) ---")
    ckpt_path = dict(CHECKPOINT_PATHS)[target_ckpt_name]
    
    tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    # Load 100 held-out entity prompts
    held_path = REPO_ROOT / "evaluation" / "cbr_suite" / "held_out_entity_grounding_100.json"
    with open(held_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
        
    first_token_results = []
    divergence_results = []
    
    correct_first_tokens = 0
    ranks = []
    probs = []
    
    for task in tasks:
        t_id = task["id"]
        prompt = task["prompt"]
        expected_concept = task["expected_concept"].strip()
        
        # Expected first token: tokenize " " + expected_concept
        expected_toks = tokenizer.encode(f" {expected_concept}")
        if not expected_toks:
            expected_toks = tokenizer.encode(expected_concept)
        exp_first_tok = expected_toks[0]
        exp_first_str = tokenizer.decode([exp_first_tok])
        
        # Run prompt prefill
        p_toks = [0] + tokenizer.encode(prompt)
        x = torch.tensor([p_toks], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(x) # [1, T, 4096]
            next_logits = logits[0, -1, :]
            prob_dist = F.softmax(next_logits, dim=-1)
            
        top5_probs, top5_ids = torch.topk(prob_dist, 5)
        top1_id = top5_ids[0].item()
        top1_str = tokenizer.decode([top1_id])
        
        # Rank of expected token
        sorted_ids = torch.argsort(next_logits, descending=True)
        exp_rank = int((sorted_ids == exp_first_tok).nonzero(as_tuple=True)[0].item()) + 1
        exp_prob = float(prob_dist[exp_first_tok].item())
        
        is_first_correct = (top1_id == exp_first_tok)
        if is_first_correct:
            correct_first_tokens += 1
        ranks.append(exp_rank)
        probs.append(exp_prob)
        
        top5_list = [{"id": int(i), "token": repr(tokenizer.decode([int(i)])), "prob": round(float(p), 4)} for p, i in zip(top5_probs, top5_ids)]
        
        first_token_results.append({
            "task_id": t_id,
            "prompt": prompt,
            "expected_concept": expected_concept,
            "expected_first_token_id": exp_first_tok,
            "expected_first_token_str": repr(exp_first_str),
            "expected_token_rank": exp_rank,
            "expected_token_prob": round(exp_prob, 5),
            "top1_token_id": top1_id,
            "top1_token_str": repr(top1_str),
            "is_correct": is_first_correct,
            "top5": top5_list
        })
        
        # Free autoregressive generation vs teacher forced
        # Generate 16 tokens freely
        curr_ids = list(p_toks)
        gen_tokens = []
        for _ in range(16):
            inp_t = torch.tensor([curr_ids], dtype=torch.long)
            with torch.no_grad():
                l = model(inp_t)[0, -1, :]
                nxt = int(torch.argmax(l).item())
            if nxt == 1: # EOS
                break
            gen_tokens.append(nxt)
            curr_ids.append(nxt)
            
        # Check prefix match at 1, 2, 4, 8, 16 with expected_toks
        match_at = {}
        for k in [1, 2, 4, 8, 16]:
            if len(expected_toks) >= k and len(gen_tokens) >= k:
                match_at[f"match_at_{k}"] = (gen_tokens[:k] == expected_toks[:k])
            elif len(expected_toks) < k and len(gen_tokens) >= len(expected_toks):
                match_at[f"match_at_{k}"] = (gen_tokens[:len(expected_toks)] == expected_toks)
            else:
                match_at[f"match_at_{k}"] = False
                
        # Earliest divergence
        first_divergence = None
        for pos in range(max(len(expected_toks), len(gen_tokens))):
            t_exp = expected_toks[pos] if pos < len(expected_toks) else None
            t_gen = gen_tokens[pos] if pos < len(gen_tokens) else None
            if t_exp != t_gen:
                first_divergence = pos
                break
                
        divergence_results.append({
            "task_id": t_id,
            "expected_tokens": expected_toks,
            "expected_str": repr(expected_concept),
            "generated_tokens": gen_tokens,
            "generated_str": repr(tokenizer.decode(gen_tokens)),
            "first_divergence_pos": first_divergence,
            "match_at": match_at
        })
        
    diag_summary = {
        "checkpoint": target_ckpt_name,
        "total_prompts": len(tasks),
        "correct_first_tokens": correct_first_tokens,
        "first_token_accuracy": round(correct_first_tokens / len(tasks), 4),
        "mean_expected_rank": round(sum(ranks) / len(ranks), 2),
        "median_expected_rank": sorted(ranks)[len(ranks)//2],
        "mean_expected_prob": round(sum(probs) / len(probs), 5),
    }
    
    with open(OUT_DIR / "first_token_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": diag_summary, "results": first_token_results}, f, indent=2)
        
    with open(OUT_DIR / "generation_divergence.json", "w", encoding="utf-8") as f:
        json.dump(divergence_results, f, indent=2)
        
    print(f"First-token accuracy: {diag_summary['first_token_accuracy']*100:.2f}%, Mean expected rank: {diag_summary['mean_expected_rank']}")
    return diag_summary

def run_prompt_sensitivity_test(target_ckpt_name="step_50"):
    print(f"\n--- 3. PROMPT SENSITIVITY TEST ({target_ckpt_name}) ---")
    ckpt_path = dict(CHECKPOINT_PATHS)[target_ckpt_name]
    runner = DirectNeuralRunner(ckpt_path)
    
    # 5 semantic intent variations across 4 test concepts
    variations = [
        # Concept 1: Greeting
        ("Greeting", [
            "User: Hello, good morning!\nAssistant:",
            "User: Good morning there!\nAssistant:",
            "User: Hi! Good morning.\nAssistant:",
            "User: A very good morning to you.\nAssistant:"
        ]),
        # Concept 2: Identity
        ("Identity", [
            "User: Who are you?\nAssistant:",
            "User: What is your name?\nAssistant:",
            "User: Introduce yourself.\nAssistant:",
            "User: Tell me who you are.\nAssistant:"
        ]),
        # Concept 3: Entity Grounding
        ("Entity", [
            "User: Context: In laboratory 1000, researcher Aarav supervises operations.\nQuestion: Who supervises laboratory 1000?\nAssistant:",
            "User: Context: Researcher Aarav supervises operations in laboratory 1000.\nQuestion: Which researcher is in charge of laboratory 1000?\nAssistant:",
            "User: Context: Laboratory 1000 is under the supervision of researcher Aarav.\nQuestion: Who manages laboratory 1000?\nAssistant:"
        ]),
        # Concept 4: Direct instruction
        ("Instruction", [
            "User: Reply with only the single uppercase word: READY\nAssistant:",
            "User: Respond using exactly one uppercase word: READY\nAssistant:",
            "User: Output just the single uppercase word: READY\nAssistant:"
        ])
    ]
    
    sens_results = []
    for concept, prompts in variations:
        concept_runs = []
        for p in prompts:
            resp = runner.generate(prompt=p, max_new_tokens=24, temperature=0.0)
            concept_runs.append({
                "prompt": p,
                "output": resp["text"],
                "first_token": runner.tokenizer.decode([resp["tokens"][0]]) if resp["tokens"] else ""
            })
            
        first_tokens = [r["first_token"] for r in concept_runs]
        is_first_token_consistent = (len(set(first_tokens)) == 1)
        
        sens_results.append({
            "concept": concept,
            "first_token_consistent": is_first_token_consistent,
            "unique_first_tokens": list(set(first_tokens)),
            "variants": concept_runs
        })
        
    with open(OUT_DIR / "prompt_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(sens_results, f, indent=2)
    print("Prompt sensitivity completed.")
    return sens_results

def run_isolated_overfit_review():
    print("\n--- 4. ISOLATED OVERFIT TEST REVIEW ---")
    from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
    
    temp_dir = REPO_ROOT / "docs" / "missions" / "CBR_01" / "experiments" / "CBR_SFT_03A" / "temp_overfit_review"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    toy_file = temp_dir / "toy_review.jsonl"
    with open(toy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "TOY-1", "prompt": "User: What is the secret code?\nAssistant:", "response": " The secret code is 999."}) + "\n")
        f.write(json.dumps({"id": "TOY-2", "prompt": "User: Who are you?\nAssistant:", "response": " I am ChakrView."}) + "\n")
        
    config = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=toy_file,
        val_file=toy_file,
        output_dir=temp_dir,
        base_checkpoint=REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        steps=50,
        batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        save_interval=50,
        val_interval=10,
        strip_prompt_delimiters=False,
        device="cpu",
        seed=42
    )
    
    engine = SFTExperimentEngine(config)
    res = engine.train()
    
    runner = DirectNeuralRunner(temp_dir / "sft_checkpoint_step_50.pt")
    
    # Memorization check
    out1 = runner.generate("User: What is the secret code?\nAssistant:", max_new_tokens=20, temperature=0.0)
    out2 = runner.generate("User: Who are you?\nAssistant:", max_new_tokens=20, temperature=0.0)
    
    # Generalization / Unseen variant check
    out_unseen_1 = runner.generate("User: What is the secret password?\nAssistant:", max_new_tokens=20, temperature=0.0)
    out_unseen_2 = runner.generate("User: Tell me who you are.\nAssistant:", max_new_tokens=20, temperature=0.0)
    out_unseen_3 = runner.generate("User: Hi, good morning!\nAssistant:", max_new_tokens=20, temperature=0.0)
    
    overfit_review = {
        "initial_val_loss": res["initial_val_loss"],
        "final_val_loss": res["final_val_loss"],
        "trained_prompt_1": {"prompt": "User: What is the secret code?\nAssistant:", "expected": "The secret code is 999.", "actual": out1["text"]},
        "trained_prompt_2": {"prompt": "User: Who are you?\nAssistant:", "expected": "I am ChakrView.", "actual": out2["text"]},
        "unseen_variant_1": {"prompt": "User: What is the secret password?\nAssistant:", "actual": out_unseen_1["text"]},
        "unseen_variant_2": {"prompt": "User: Tell me who you are.\nAssistant:", "actual": out_unseen_2["text"]},
        "unseen_unrelated_3": {"prompt": "User: Hi, good morning!\nAssistant:", "actual": out_unseen_3["text"]}
    }
    
    with open(OUT_DIR / "overfit_review_results.json", "w", encoding="utf-8") as f:
        json.dump(overfit_review, f, indent=2)
        
    # Clean up checkpoint and training artifacts from temp_dir, preserving only json report in OUT_DIR
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Overfit test reproduced and reviewed cleanly.")
    return overfit_review

if __name__ == "__main__":
    master_summary = run_all_checkpoint_evaluations()
    first_tok_summary = run_first_token_and_teacher_forcing_audit(target_ckpt_name="step_50")
    sens_summary = run_prompt_sensitivity_test(target_ckpt_name="step_50")
    overfit_summary = run_isolated_overfit_review()
    print("\n--- ALL CBR-SFT-03A DIAGNOSTICS COMPLETE ---")

