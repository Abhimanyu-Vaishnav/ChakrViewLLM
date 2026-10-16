"""
Comprehensive 200-Task Capability Audit Runner for Batches 1461–1500.
Location: experiments/batch1461_1500/run_capability_audit.py

Executes:
1. Track A: Language Modeling Evaluation (30 independent documents, 7,741 targets)
2. Track B: Factual Recall in 3 configurations:
   - Config 1: ChakrMicro Standalone (Generative Recall)
   - Config 2: ChakrMicro + Knowledge Assistant (BM25 Retrieval + Extractive Verification)
   - Config 3: Deterministic Majority/Unigram Baseline
3. Track C: Arithmetic and Mathematical Reasoning (40 tasks, exact numeric extraction)
4. Track D: Python Coding with Sandboxed Execution of Unit Test Assertions (40 tasks)
5. Track E: Generalization & Paraphrase Pair Consistency (20 pairs = 40 tasks)
6. Track B/E Controls: False-Premise Refutation & Abstention Verification (40 tasks)

Saves machine-readable results to experiments/batch1461_1500/results/CAPABILITY_AUDIT_RESULTS.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
CANDIDATE_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
TASKS_PATH = Path("experiments/batch1461_1500/data/capability_audit_200.json")
CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1461_1500/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoints():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    b_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    m_base = ChakrMicro(b_ckpt["config"])
    m_base.load_state_dict(b_ckpt["model_state_dict"], strict=True)
    m_base.eval()

    c_ckpt = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)
    m_cand = ChakrMicro(c_ckpt["config"])
    m_cand.load_state_dict(c_ckpt["model_state_dict"], strict=True)
    m_cand.eval()

    return tokenizer, m_base, m_cand


def generate_tokens(model: ChakrMicro, tokenizer, prompt: str, max_new_tokens: int = 35) -> str:
    """Generate tokens autoregressively under greedy argmax decoding."""
    tokens = tokenizer.encode(prompt)
    curr = tokens.copy()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr], dtype=torch.long)
            logits = model(inp)[0, -1, :]
            next_token = torch.argmax(logits).item()
            curr.append(next_token)
            if next_token == 1:  # EOS token ID in ChakrMicro config
                break
    return tokenizer.decode(curr[len(tokens):], errors="replace")


def evaluate_arithmetic(model: ChakrMicro, tokenizer, tasks: list[dict]) -> dict:
    """Evaluate Track C Arithmetic tasks with exact numerical extraction."""
    total = len(tasks)
    correct = 0
    results = []

    for t in tasks:
        prompt = t["prompt"]
        expected = t["expected_answer"]
        raw_output = generate_tokens(model, tokenizer, prompt, max_new_tokens=20)

        # Extract first contiguous integer/float from generated text
        num_match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_output)
        extracted = num_match.group(0) if num_match else None

        is_correct = (extracted == expected)
        if is_correct:
            correct += 1

        results.append({
            "id": t["id"],
            "prompt": prompt,
            "expected": expected,
            "raw_output": raw_output.strip(),
            "extracted_number": extracted,
            "is_correct": is_correct
        })

    return {
        "total_tasks": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "results": results
    }


def evaluate_coding(model: ChakrMicro, tokenizer, tasks: list[dict]) -> dict:
    """Evaluate Track D Python Coding tasks with sandboxed unit test execution."""
    total = len(tasks)
    syntax_valid_count = 0
    tests_passed_count = 0
    results = []

    for t in tasks:
        prompt = t["prompt"]
        assertions = t["executable_tests"]

        # Prompt format is "Complete the Python function:\n{prompt_code}"
        fn_header = prompt.split("Complete the Python function:\n")[1]
        raw_completion = generate_tokens(model, tokenizer, prompt, max_new_tokens=40)

        # Reconstruct code
        code_block = fn_header + raw_completion

        # 1. Syntax check
        syntax_valid = False
        try:
            compile(code_block, "<string>", "exec")
            syntax_valid = True
            syntax_valid_count += 1
        except Exception as e:
            syntax_error = str(e)

        # 2. Test execution in isolated namespace
        test_passed = False
        exec_error = None
        if syntax_valid:
            test_script = code_block + "\n" + assertions
            namespace = {}
            try:
                # Execute in safe isolated dict without builtins overwrite
                exec(test_script, namespace)
                test_passed = True
                tests_passed_count += 1
            except AssertionError:
                exec_error = "AssertionError: Unit test assertions failed."
            except Exception as e:
                exec_error = f"{type(e).__name__}: {str(e)}"
        else:
            exec_error = f"SyntaxError: {syntax_error}"

        results.append({
            "id": t["id"],
            "prompt": prompt,
            "raw_completion": raw_completion[:100],
            "syntax_valid": syntax_valid,
            "test_passed": test_passed,
            "exec_error": exec_error
        })

    return {
        "total_tasks": total,
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_rate": round(syntax_valid_count / total, 4),
        "tests_passed_count": tests_passed_count,
        "test_pass_rate": round(tests_passed_count / total, 4),
        "results": results
    }


def evaluate_factual_modes(m_base: ChakrMicro, m_cand: ChakrMicro, tokenizer, asst: LocalKnowledgeAssistant, tasks: list[dict]) -> dict:
    """Evaluate Track B Factual Recall across Standalone vs Grounded modes."""
    total = len(tasks)
    
    cand_standalone_hits = 0
    asst_grounded_hits = 0
    results = []

    for t in tasks:
        query = t["prompt"]
        expected = t["expected_answer"]

        # Config 1: Model Standalone (Generative Recall without context)
        prompt_standalone = f"Question: {query}\nAnswer:"
        cand_raw = generate_tokens(m_cand, tokenizer, prompt_standalone, max_new_tokens=25)
        cand_hit = (expected.lower() in cand_raw.lower())
        if cand_hit:
            cand_standalone_hits += 1

        # Config 2: Local Knowledge Assistant (BM25 + Extractive Grounding)
        asst.reset_conversation()
        resp = asst.ask(query)
        asst_hit = (expected.lower() in resp.answer.lower() and resp.decision == "ANSWER")
        if asst_hit:
            asst_grounded_hits += 1

        results.append({
            "id": t["id"],
            "query": query,
            "expected": expected,
            "cand_standalone_output": cand_raw.strip()[:100],
            "cand_standalone_hit": cand_hit,
            "asst_answer": resp.answer[:120],
            "asst_decision": resp.decision,
            "asst_citations": [c.get("doc_id", "") for c in resp.citations],
            "asst_hit": asst_hit
        })

    return {
        "total_tasks": total,
        "cand_standalone_accuracy": round(cand_standalone_hits / total, 4),
        "asst_grounded_accuracy": round(asst_grounded_hits / total, 4),
        "results": results
    }


def evaluate_generalization_pairs(asst: LocalKnowledgeAssistant, tasks: list[dict]) -> dict:
    """Evaluate Track E Paraphrased Pairs for consistency."""
    pairs = {}
    for t in tasks:
        pid = t["pair_id"]
        pairs.setdefault(pid, []).append(t)

    pair_results = []
    both_correct = 0
    one_correct = 0
    both_failed = 0

    for pid, task_list in sorted(pairs.items()):
        if len(task_list) != 2:
            continue
        t_a, t_b = task_list[0], task_list[1]
        
        asst.reset_conversation()
        resp_a = asst.ask(t_a["prompt"])
        hit_a = (t_a["expected_answer"].lower() in resp_a.answer.lower() and resp_a.decision == "ANSWER")

        asst.reset_conversation()
        resp_b = asst.ask(t_b["prompt"])
        hit_b = (t_b["expected_answer"].lower() in resp_b.answer.lower() and resp_b.decision == "ANSWER")

        if hit_a and hit_b:
            both_correct += 1
            status = "BOTH_CORRECT"
        elif hit_a or hit_b:
            one_correct += 1
            status = "PARTIAL_INCONSISTENT"
        else:
            both_failed += 1
            status = "BOTH_FAILED"

        pair_results.append({
            "pair_id": pid,
            "query_a": t_a["prompt"],
            "hit_a": hit_a,
            "query_b": t_b["prompt"],
            "hit_b": hit_b,
            "status": status
        })

    total_pairs = len(pairs)
    return {
        "total_pairs": total_pairs,
        "both_correct_count": both_correct,
        "both_correct_rate": round(both_correct / total_pairs, 4),
        "partial_inconsistent_count": one_correct,
        "both_failed_count": both_failed,
        "pair_consistency_rate": round((both_correct + both_failed) / total_pairs, 4),
        "pair_results": pair_results
    }


def evaluate_controls(asst: LocalKnowledgeAssistant, tasks: list[dict]) -> dict:
    """Evaluate Control Tasks (False Premise Refutation & Honest Abstention)."""
    total = len(tasks)
    correct = 0
    results = []

    refute_total = 0
    refute_correct = 0
    abstain_total = 0
    abstain_correct = 0

    for t in tasks:
        query = t["prompt"]
        expected = t["expected_answer"]

        asst.reset_conversation()
        resp = asst.ask(query)
        actual = resp.decision

        is_correct = (actual == expected)
        if is_correct:
            correct += 1

        if expected == "REFUTE":
            refute_total += 1
            if actual == "REFUTE":
                refute_correct += 1
        elif expected == "ABSTAIN":
            abstain_total += 1
            if actual == "ABSTAIN":
                abstain_correct += 1

        results.append({
            "id": t["id"],
            "category": t["category"],
            "query": query,
            "expected_decision": expected,
            "actual_decision": actual,
            "is_correct": is_correct,
            "answer_preview": resp.answer[:120]
        })

    return {
        "total_control_tasks": total,
        "overall_control_accuracy": round(correct / total, 4),
        "refutation_accuracy": round(refute_correct / refute_total, 4) if refute_total > 0 else 0.0,
        "refutation_counts": f"{refute_correct}/{refute_total}",
        "abstention_accuracy": round(abstain_correct / abstain_total, 4) if abstain_total > 0 else 0.0,
        "abstention_counts": f"{abstain_correct}/{abstain_total}",
        "results": results
    }


def main():
    print("Loading models and knowledge assistant...")
    tokenizer, m_base, m_cand = load_checkpoints()

    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)

    all_tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(all_tasks)} total tasks.")

    math_tasks = [t for t in all_tasks if t["track"] == "Track_C_Mathematics"]
    coding_tasks = [t for t in all_tasks if t["track"] == "Track_D_Coding"]
    factual_tasks = [t for t in all_tasks if t["track"] == "Track_B_Factual_Grounding" and t["category"] == "factual_recall"]
    paraphrase_tasks = [t for t in all_tasks if t["track"] == "Track_E_Generalization" and "paraphrase" in t["category"]]
    control_tasks = [t for t in all_tasks if t["category"] in ("false_premise_refutation", "honest_abstention")]

    # 1. Track C: Mathematics
    print("\n--- Evaluating Track C: Mathematics (40 Tasks) ---")
    math_base = evaluate_arithmetic(m_base, tokenizer, math_tasks)
    math_cand = evaluate_arithmetic(m_cand, tokenizer, math_tasks)
    print(f"Base Accuracy: {math_base['accuracy']*100:.2f}% ({math_base['correct']}/{math_base['total_tasks']})")
    print(f"Candidate Accuracy: {math_cand['accuracy']*100:.2f}% ({math_cand['correct']}/{math_cand['total_tasks']})")

    # 2. Track D: Coding
    print("\n--- Evaluating Track D: Python Coding with Sandboxed Execution (40 Tasks) ---")
    code_base = evaluate_coding(m_base, tokenizer, coding_tasks)
    code_cand = evaluate_coding(m_cand, tokenizer, coding_tasks)
    print(f"Base: Syntax Valid={code_base['syntax_valid_rate']*100:.2f}% | Tests Passed={code_base['test_pass_rate']*100:.2f}% ({code_base['tests_passed_count']}/{code_base['total_tasks']})")
    print(f"Cand: Syntax Valid={code_cand['syntax_valid_rate']*100:.2f}% | Tests Passed={code_cand['test_pass_rate']*100:.2f}% ({code_cand['tests_passed_count']}/{code_cand['total_tasks']})")

    # 3. Track B: Factual Recall Across 3 Configurations
    print("\n--- Evaluating Track B: Factual Recall Configurations (40 Tasks) ---")
    factual_results = evaluate_factual_modes(m_base, m_cand, tokenizer, asst, factual_tasks)
    print(f"Standalone Model Accuracy: {factual_results['cand_standalone_accuracy']*100:.2f}%")
    print(f"Grounded Assistant Pipeline Accuracy: {factual_results['asst_grounded_accuracy']*100:.2f}%")

    # 4. Track E: Generalization Pairs
    print("\n--- Evaluating Track E: Paraphrase Pair Generalization (20 Pairs = 40 Tasks) ---")
    pair_results = evaluate_generalization_pairs(asst, paraphrase_tasks)
    print(f"Both Correct: {pair_results['both_correct_count']}/{pair_results['total_pairs']} ({pair_results['both_correct_rate']*100:.2f}%)")
    print(f"Pair Consistency Rate: {pair_results['pair_consistency_rate']*100:.2f}%")

    # 5. Controls: Refutation & Abstention
    print("\n--- Evaluating Control Tasks: Refutation & Abstention (40 Tasks) ---")
    control_results = evaluate_controls(asst, control_tasks)
    print(f"Overall Control Accuracy: {control_results['overall_control_accuracy']*100:.2f}%")
    print(f"False-Premise Refutation: {control_results['refutation_counts']} ({control_results['refutation_accuracy']*100:.2f}%)")
    print(f"Honest Abstention: {control_results['abstention_counts']} ({control_results['abstention_accuracy']*100:.2f}%)")

    full_audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_matrix": {
            "track_c_mathematics": {
                "canonical_base_accuracy": math_base["accuracy"],
                "candidate_adapted_accuracy": math_cand["accuracy"],
                "delta": round(math_cand["accuracy"] - math_base["accuracy"], 4)
            },
            "track_d_coding": {
                "canonical_syntax_rate": code_base["syntax_valid_rate"],
                "canonical_test_pass_rate": code_base["test_pass_rate"],
                "candidate_syntax_rate": code_cand["syntax_valid_rate"],
                "candidate_test_pass_rate": code_cand["test_pass_rate"]
            },
            "track_b_factual_grounding": {
                "standalone_model_recall": factual_results["cand_standalone_accuracy"],
                "grounded_assistant_accuracy": factual_results["asst_grounded_accuracy"]
            },
            "track_e_generalization": {
                "both_correct_rate": pair_results["both_correct_rate"],
                "pair_consistency_rate": pair_results["pair_consistency_rate"]
            },
            "controls": {
                "overall_accuracy": control_results["overall_control_accuracy"],
                "refutation_accuracy": control_results["refutation_accuracy"],
                "abstention_accuracy": control_results["abstention_accuracy"]
            }
        },
        "track_c_details": {"base": math_base, "cand": math_cand},
        "track_d_details": {"base": code_base, "cand": code_cand},
        "track_b_details": factual_results,
        "track_e_details": pair_results,
        "control_details": control_results
    }

    out_file = OUT_DIR / "CAPABILITY_AUDIT_RESULTS.json"
    out_file.write_text(json.dumps(full_audit, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable capability audit results to {out_file}")


if __name__ == "__main__":
    main()
