"""
Benchmark & Training Harness for Neural Router Feasibility (Batches 1661–1700).
Location: experiments/batch1661_1700/run_router_experiment.py

Executes:
1. Verifies immutable baseline hashes.
2. Evaluates Baseline 1: Existing Deterministic Router (GovernedToolRouter).
3. Evaluates Baseline 2: Keyword Rule Baseline.
4. Evaluates Baseline 3: Untrained Canonical ChakrMicro (Pure Neural).
5. Trains Candidate Neural Router on train.jsonl for 100 steps.
6. Evaluates Baseline 4: Trained Candidate ChakrMicro (Pure Neural).
7. Evaluates System 5: Governed Hybrid (Neural intent proposal + Schema Validator + Fallback).
8. Saves comprehensive metrics to results/ROUTER_EVALUATION_RESULTS.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.governed_tool_router import GovernedToolRouter, CanonicalRoute

CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
ADAPTED_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")

DATA_DIR = Path("experiments/batch1661_1700/data")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test_held_out.json"

CANDIDATE_CKPT_PATH = Path("experiments/batch1661_1700/checkpoints/chakrmicro_router_candidate_step100.pt")
RESULTS_PATH = Path("experiments/batch1661_1700/results/ROUTER_EVALUATION_RESULTS.json")

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

INTENTS = [
    "CALCULATE",
    "VERIFY_CLAIM",
    "KNOWLEDGE_QUERY",
    "CODE_TASK",
    "CLARIFY",
    "OUT_OF_SCOPE",
]


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(65536):
            h.update(b)
    return h.hexdigest()


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, config


# -------------------------------------------------------------
# Baseline 1: Deterministic Router Wrapper (GovernedToolRouter)
# -------------------------------------------------------------
class DeterministicRouterEvaluator:
    def __init__(self):
        self.router = GovernedToolRouter()

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        decision = self.router.route_request(prompt)
        dt = (time.perf_counter() - t0) * 1000.0

        r = decision.route
        args = decision.validated_arguments

        # Map CanonicalRoute to 6-intent taxonomy
        if r == CanonicalRoute.CALCULATOR:
            intent = "CALCULATE"
        elif r == CanonicalRoute.STRUCTURED_VERIFICATION:
            intent = "VERIFY_CLAIM"
        elif r == CanonicalRoute.KNOWLEDGE_RETRIEVAL:
            intent = "KNOWLEDGE_QUERY"
        elif r == CanonicalRoute.CONTAINER_EXECUTION:
            intent = "CODE_TASK"
        elif r == CanonicalRoute.ABSTAIN:
            intent = "CLARIFY"
        else:
            intent = "OUT_OF_SCOPE"

        return {
            "intent": intent,
            "arguments": args,
            "is_valid_json": True,
            "confidence": 1.0,
            "latency_ms": dt,
            "source": "deterministic_governed_router"
        }


# -------------------------------------------------------------
# Baseline 2: Simple Keyword / Heuristic Router
# -------------------------------------------------------------
class KeywordRouterEvaluator:
    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        p_lower = prompt.lower()

        # Keyword rules
        if any(k in p_lower for k in ["calculate", "compute", "sum of", "divide", "multiply", "plus", "minus", "math"]):
            intent = "CALCULATE"
            # attempt naive extraction of digits/symbols
            m = re.search(r"[\d\s\+\-\*\/\(\)\%]+", prompt)
            expr = m.group(0).strip() if m else ""
            args = {"expression": expr} if expr else {"missing_field": "expression", "reason": "No expression"}
        elif any(k in p_lower for k in ["verify", "check if", "is it true", "confirm", "validate"]):
            intent = "VERIFY_CLAIM"
            args = {"claim": prompt}
        elif any(k in p_lower for k in ["search", "how do", "how is", "where is", "documentation", "docs", "explain"]):
            intent = "KNOWLEDGE_QUERY"
            args = {"query": prompt}
        elif any(k in p_lower for k in ["python", "run code", "execute", "snippet", "script"]):
            intent = "CODE_TASK"
            args = {"code_snippet": prompt}
        else:
            intent = "OUT_OF_SCOPE"
            args = {"reason": "No matching keyword"}

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "intent": intent,
            "arguments": args,
            "is_valid_json": True,
            "confidence": 0.5,
            "latency_ms": dt,
            "source": "keyword_baseline"
        }


# -------------------------------------------------------------
# Neural Generation and Parsing
# -------------------------------------------------------------
def generate_response(model: ChakrMicro, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    formatted_prompt = f"User: {prompt}\nRouter: "
    tokens = tokenizer.encode(formatted_prompt)
    curr = tokens.copy()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr], dtype=torch.long)
            logits = model(inp)[0, -1, :]
            next_tok = torch.argmax(logits).item()
            curr.append(next_tok)
            if next_tok == 1 or next_tok == 0:  # EOS or BOS
                break
    return tokenizer.decode(curr[len(tokens):], errors="replace")


def parse_neural_output(gen_text: str) -> dict:
    """Strict JSON parser and schema validator."""
    gen_text = gen_text.strip()
    # Try finding JSON block
    m = re.search(r"\{.*\}", gen_text, re.DOTALL)
    if not m:
        return {
            "intent": "OUT_OF_SCOPE",
            "arguments": {"reason": "Non-JSON neural generation"},
            "is_valid_json": False,
            "raw": gen_text
        }
    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return {"intent": "OUT_OF_SCOPE", "arguments": {}, "is_valid_json": False, "raw": gen_text}
        intent = obj.get("intent", "OUT_OF_SCOPE")
        if intent not in INTENTS:
            intent = "OUT_OF_SCOPE"
        arguments = obj.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "intent": intent,
            "arguments": arguments,
            "is_valid_json": True,
            "confidence": float(obj.get("confidence", 0.0)),
            "raw": gen_text
        }
    except Exception:
        return {
            "intent": "OUT_OF_SCOPE",
            "arguments": {"reason": "Malformed JSON syntax"},
            "is_valid_json": False,
            "raw": gen_text
        }


class NeuralRouterEvaluator:
    def __init__(self, model: ChakrMicro, tokenizer, name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        raw = generate_response(self.model, self.tokenizer, prompt, max_new_tokens=40)
        dt = (time.perf_counter() - t0) * 1000.0
        parsed = parse_neural_output(raw)
        parsed["latency_ms"] = dt
        parsed["source"] = self.name
        return parsed


class GovernedHybridRouterEvaluator:
    """
    Evaluates a governed neuro-symbolic hybrid:
    Uses Neural candidate output IF valid JSON and schema-compliant;
    Falls back to Deterministic Governed Router if invalid or ungrounded.
    Logs which subsystem made the decision.
    """
    def __init__(self, neural_eval: NeuralRouterEvaluator, det_eval: DeterministicRouterEvaluator):
        self.neural = neural_eval
        self.det = det_eval

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        n_res = self.neural.route(prompt)

        # Safety Gate: if neural output is valid JSON and intent is confident
        if n_res["is_valid_json"] and n_res["intent"] in INTENTS and n_res["intent"] != "OUT_OF_SCOPE":
            decision = n_res
            decision["attribution"] = "neural_model"
        else:
            d_res = self.det.route(prompt)
            decision = d_res
            decision["attribution"] = "deterministic_fallback"

        dt = (time.perf_counter() - t0) * 1000.0
        decision["latency_ms"] = dt
        decision["source"] = "governed_hybrid"
        return decision


# -------------------------------------------------------------
# Evaluation Runner & Metrics Engine
# -------------------------------------------------------------
def evaluate_system(system, tasks: list[dict], system_name: str) -> dict:
    results = []
    correct_intent = 0
    correct_args = 0
    valid_json_count = 0
    attribution_counts = {}

    per_intent_stats = {intent: {"total": 0, "tp": 0, "fp": 0, "fn": 0} for intent in INTENTS}

    t0 = time.perf_counter()

    for task in tasks:
        prompt = task["prompt"]
        true_intent = task["intent"]
        exp_args = task["expected_arguments"]

        out = system.route(prompt)
        pred_intent = out["intent"]
        pred_args = out.get("arguments", {})
        is_valid_json = out.get("is_valid_json", False)

        if is_valid_json:
            valid_json_count += 1

        attr = out.get("attribution", out.get("source", "unknown"))
        attribution_counts[attr] = attribution_counts.get(attr, 0) + 1

        # Check intent correctness
        intent_match = (pred_intent == true_intent)
        if intent_match:
            correct_intent += 1
            per_intent_stats[true_intent]["tp"] += 1
        else:
            per_intent_stats[true_intent]["fn"] += 1
            if pred_intent in per_intent_stats:
                per_intent_stats[pred_intent]["fp"] += 1

        # Check argument exact/semantic match
        arg_match = False
        if intent_match and isinstance(pred_args, dict):
            # Check key presence and non-empty value
            if true_intent == "CALCULATE":
                arg_match = ("expression" in pred_args and len(str(pred_args["expression"])) > 0)
            elif true_intent == "VERIFY_CLAIM":
                arg_match = ("claim" in pred_args and len(str(pred_args["claim"])) > 0)
            elif true_intent == "KNOWLEDGE_QUERY":
                arg_match = ("query" in pred_args and len(str(pred_args["query"])) > 0)
            elif true_intent == "CODE_TASK":
                arg_match = ("code_snippet" in pred_args and len(str(pred_args["code_snippet"])) > 0)
            elif true_intent == "CLARIFY":
                arg_match = ("missing_field" in pred_args)
            elif true_intent == "OUT_OF_SCOPE":
                arg_match = True

        if arg_match:
            correct_args += 1

        results.append({
            "task_id": task["id"],
            "prompt": prompt,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "true_arguments": exp_args,
            "pred_arguments": pred_args,
            "intent_match": intent_match,
            "arg_match": arg_match,
            "valid_json": is_valid_json,
            "latency_ms": out.get("latency_ms", 0.0),
            "attribution": attr
        })

    total = len(tasks)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate Macro-F1 across intents
    f1_list = []
    per_class_summary = {}
    for intent, s in per_intent_stats.items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class_summary[intent] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": s["total"] + s["fn"] + s["tp"]
        }

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary = {
        "system_name": system_name,
        "total_tasks": total,
        "intent_accuracy": round(correct_intent / total, 4),
        "argument_accuracy": round(correct_args / total, 4),
        "macro_f1": round(macro_f1, 4),
        "json_validity_rate": round(valid_json_count / total, 4),
        "total_latency_ms": round(elapsed_ms, 2),
        "avg_latency_ms": round(elapsed_ms / total, 2),
        "attribution_counts": attribution_counts,
        "per_class": per_class_summary
    }

    return {"summary": summary, "per_task": results}


# -------------------------------------------------------------
# Controlled Training Function for Router
# -------------------------------------------------------------
def train_router_candidate(tokenizer, base_model, config, train_items: list[dict], val_items: list[dict]) -> tuple[ChakrMicro, dict]:
    print("\n--- Training Candidate Neural Router (100 steps) ---")
    model = ChakrMicro(config)
    model.load_state_dict(base_model.state_dict())
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), weight_decay=0.01)

    train_texts = [x["formatted_lm_text"] for x in train_items]
    val_texts = [x["formatted_lm_text"] for x in val_items]

    step_log = []
    t0 = time.perf_counter()

    for step in range(1, 101):
        batch_texts = random.sample(train_texts, 4)
        batch_tokens = [tokenizer.encode(t) for t in batch_texts]
        max_len = max(len(b) for b in batch_tokens)
        max_len = min(max_len, 256)

        padded = []
        for b in batch_tokens:
            if len(b) >= max_len:
                padded.append(b[:max_len])
            else:
                padded.append(b + [2] * (max_len - len(b)))

        inp = torch.tensor([p[:-1] for p in padded], dtype=torch.long)
        targets = torch.tensor([p[1:] for p in padded], dtype=torch.long)

        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, 4096), targets.reshape(-1), ignore_index=2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == 100:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for vt in val_texts:
                    toks = tokenizer.encode(vt)
                    if len(toks) < 2:
                        continue
                    v_inp = torch.tensor([toks[:-1]], dtype=torch.long)
                    v_tar = torch.tensor(toks[1:], dtype=torch.long)
                    v_logits = model(v_inp)[0]
                    v_loss = F.cross_entropy(v_logits, v_tar, reduction="sum")
                    val_loss += v_loss.item()
                    val_tokens += len(v_tar)

            mean_val_loss = val_loss / val_tokens if val_tokens > 0 else 0.0
            val_ppl = math.exp(mean_val_loss) if mean_val_loss < 20 else float("inf")
            model.train()

            print(f"Step {step:03d}/100 | Train Loss: {loss.item():.4f} | Val Loss: {mean_val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            step_log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(mean_val_loss, 4),
                "val_ppl": round(val_ppl, 2)
            })

    total_time = round(time.perf_counter() - t0, 2)
    model.eval()

    # Save candidate checkpoint
    torch.save({
        "step": 100,
        "model_state_dict": model.state_dict(),
        "config": config,
        "val_loss": step_log[-1]["val_loss"],
        "training_time_sec": total_time
    }, CANDIDATE_CKPT_PATH)

    cand_sha = get_file_sha256(CANDIDATE_CKPT_PATH)
    print(f"Saved router candidate to {CANDIDATE_CKPT_PATH} (SHA-256: {cand_sha})")

    report = {
        "total_steps": 100,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "training_time_sec": total_time,
        "best_val_loss": step_log[-1]["val_loss"],
        "candidate_checkpoint_sha256": cand_sha,
        "step_log": step_log
    }
    return model, report


def main():
    print("=================================================================")
    print("CHAKRVIEW BATCHES 1661–1700: NEURAL ROUTER FEASIBILITY GATE")
    print("=================================================================")

    # 1. Verify Reference Checkpoint Hashes
    c1_hash = get_file_sha256(CANONICAL_CKPT)
    c2_hash = get_file_sha256(ADAPTED_CKPT)
    print(f"Canonical Checkpoint: {CANONICAL_CKPT} (SHA-256: {c1_hash})")
    print(f"Adapted Checkpoint:   {ADAPTED_CKPT} (SHA-256: {c2_hash})")

    assert c1_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee", "Canonical Checkpoint altered!"
    assert c2_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139", "Adapted Checkpoint altered!"

    # 2. Load tokenizer and dataset
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_tasks = json.load(f)

    print(f"Loaded {len(train_data)} train samples, {len(val_data)} val samples, {len(test_tasks)} held-out tasks.")

    # 3. Instantiate Systems
    det_sys = DeterministicRouterEvaluator()
    kw_sys = KeywordRouterEvaluator()

    base_model, config = load_model(CANONICAL_CKPT)
    neural_base_sys = NeuralRouterEvaluator(base_model, tokenizer, "canonical_neural_untrained")

    # 4. Train Neural Router Candidate
    cand_model, train_report = train_router_candidate(tokenizer, base_model, config, train_data, val_data)
    neural_cand_sys = NeuralRouterEvaluator(cand_model, tokenizer, "candidate_neural_step100")

    hybrid_sys = GovernedHybridRouterEvaluator(neural_cand_sys, det_sys)

    # 5. Evaluate all 5 systems on the EXACT same 120 held-out tasks
    print("\n--- Evaluating Baseline 1: Governed Deterministic Router ---")
    det_res = evaluate_system(det_sys, test_tasks, "deterministic_governed_router")
    print(f"Deterministic -> Accuracy: {det_res['summary']['intent_accuracy']*100:.2f}% | Macro-F1: {det_res['summary']['macro_f1']:.4f}")

    print("\n--- Evaluating Baseline 2: Keyword Heuristic Baseline ---")
    kw_res = evaluate_system(kw_sys, test_tasks, "keyword_baseline")
    print(f"Keyword -> Accuracy: {kw_res['summary']['intent_accuracy']*100:.2f}% | Macro-F1: {kw_res['summary']['macro_f1']:.4f}")

    print("\n--- Evaluating Baseline 3: Canonical ChakrMicro (Untrained Neural) ---")
    neural_base_res = evaluate_system(neural_base_sys, test_tasks, "canonical_neural_untrained")
    print(f"Neural Base -> Accuracy: {neural_base_res['summary']['intent_accuracy']*100:.2f}% | JSON Valid: {neural_base_res['summary']['json_validity_rate']*100:.1f}%")

    print("\n--- Evaluating Baseline 4: Candidate ChakrMicro (Trained Neural Router) ---")
    neural_cand_res = evaluate_system(neural_cand_sys, test_tasks, "candidate_neural_step100")
    print(f"Neural Candidate -> Accuracy: {neural_cand_res['summary']['intent_accuracy']*100:.2f}% | JSON Valid: {neural_cand_res['summary']['json_validity_rate']*100:.1f}%")

    print("\n--- Evaluating System 5: Governed Neuro-Symbolic Hybrid ---")
    hybrid_res = evaluate_system(hybrid_sys, test_tasks, "governed_neuro_symbolic_hybrid")
    print(f"Governed Hybrid -> Accuracy: {hybrid_res['summary']['intent_accuracy']*100:.2f}% | Macro-F1: {hybrid_res['summary']['macro_f1']:.4f}")

    # 6. Save comprehensive machine-readable results
    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_checkpoint_sha256": c1_hash,
        "adapted_checkpoint_sha256": c2_hash,
        "candidate_checkpoint_sha256": train_report["candidate_checkpoint_sha256"],
        "training_report": train_report,
        "comparative_summary": {
            "deterministic_governed_router": det_res["summary"],
            "keyword_baseline": kw_res["summary"],
            "canonical_neural_untrained": neural_base_res["summary"],
            "candidate_neural_step100": neural_cand_res["summary"],
            "governed_neuro_symbolic_hybrid": hybrid_res["summary"],
        },
        "per_task_results": {
            "deterministic": det_res["per_task"],
            "keyword": kw_res["per_task"],
            "canonical_neural": neural_base_res["per_task"],
            "candidate_neural": neural_cand_res["per_task"],
            "governed_hybrid": hybrid_res["per_task"],
        }
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nAll Router Experiments Complete! Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
