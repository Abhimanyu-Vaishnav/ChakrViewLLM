"""
Comprehensive Reproduction, Multi-Seed Training & Semantic Evaluation Engine.
Location: experiments/batch1741_1780/run_batch1741_experiments.py

Executes:
1. WP-3 Reproduction: Trains Variant B across 3 random seeds (42, 123, 999) on train.jsonl.
   Computes mean, std, Brier score, and per-seed held-out generalization metrics.
2. WP-4 & WP-5 Evaluation on 300-Task Frozen Benchmark:
   Evaluates across 6 systems:
   - Majority Baseline
   - Keyword Heuristic Baseline
   - Governed Deterministic Router
   - Variant A: Frozen Backbone + Linear Head
   - Variant B: Fine-Tuned Backbone + Linear Head (Best Seed Checkpoint)
   - Governed Neuro-Symbolic Hybrid
3. Safety Metrics:
   - Separates Attempted Routes, Rejected Routes, Accepted Routes, and Actually Executed Tools.
   - Evaluates Hard-Negative tool mentions (20 tasks) and Adversarial Payloads (10 tasks).
4. Saves full results to results/BENCHMARK_300_EVALUATION_RESULTS.json.
"""
from __future__ import annotations

import copy
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
from chakrview.brain.normalization import RMSNorm
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.governed_tool_router import GovernedToolRouter, CanonicalRoute

CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
ADAPTED_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")

DATA_DIR = Path("experiments/batch1741_1780/data")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_300_FILE = DATA_DIR / "test_held_out_300.json"

CHECKPOINTS_DIR = Path("experiments/batch1741_1780/checkpoints")
RESULTS_PATH = Path("experiments/batch1741_1780/results/BENCHMARK_300_EVALUATION_RESULTS.json")

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


class ChakrMicroIntentClassifier(nn.Module):
    def __init__(self, backbone: ChakrMicro, num_classes: int = 6, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.classifier = nn.Linear(self.config.d_model, num_classes, bias=True)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, h = self.backbone(input_ids, attention_mask=attention_mask, return_hidden_states=True)
        if attention_mask is not None:
            lens = attention_mask.sum(dim=-1).long() - 1
            lens = torch.clamp(lens, min=0)
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            h_pooled = h[batch_indices, lens, :]
        else:
            h_pooled = h[:, -1, :]

        normed = self.norm(h_pooled)
        return self.classifier(normed)


def load_canonical_model():
    ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, config


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------
def train_single_seed(
    seed: int,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    freeze_backbone: bool = False,
    steps: int = 100,
    lr_head: float = 1e-4,
    lr_backbone: float = 3e-5,
    name: str = "classifier"
) -> tuple[ChakrMicroIntentClassifier, dict]:
    torch.manual_seed(seed)
    random.seed(seed)

    base_m, _ = load_canonical_model()
    classifier = ChakrMicroIntentClassifier(base_m, num_classes=6, freeze_backbone=freeze_backbone)

    params = []
    if not freeze_backbone:
        params.append({"params": classifier.backbone.parameters(), "lr": lr_backbone})
    params.append({"params": classifier.norm.parameters(), "lr": lr_head})
    params.append({"params": classifier.classifier.parameters(), "lr": lr_head})

    optimizer = AdamW(params, weight_decay=0.01)
    classifier.train()

    t0 = time.perf_counter()
    step_log = []

    for step in range(1, steps + 1):
        batch = random.sample(train_data, min(8, len(train_data)))
        token_lists = [tokenizer.encode(b["input_text"]) for b in batch]
        max_l = min(max(len(t) for t in token_lists), 128)

        padded_toks = []
        mask_list = []
        for t in token_lists:
            t = t[:max_l]
            pad_len = max_l - len(t)
            padded_toks.append(t + [2] * pad_len)
            mask_list.append([1] * len(t) + [0] * pad_len)

        inp = torch.tensor(padded_toks, dtype=torch.long)
        mask = torch.tensor(mask_list, dtype=torch.float32)
        labels = torch.tensor([b["intent_id"] for b in batch], dtype=torch.long)

        logits = classifier(inp, attention_mask=mask)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for v in val_data:
                    v_toks = tokenizer.encode(v["input_text"])[:128]
                    v_inp = torch.tensor([v_toks], dtype=torch.long)
                    v_mask = torch.tensor([[1] * len(v_toks)], dtype=torch.float32)
                    v_label = torch.tensor([v["intent_id"]], dtype=torch.long)
                    v_logit = classifier(v_inp, attention_mask=v_mask)
                    v_loss = F.cross_entropy(v_logit, v_label)
                    val_loss += v_loss.item()
                    if torch.argmax(v_logit, dim=-1).item() == v["intent_id"]:
                        val_correct += 1

            mean_val_loss = val_loss / len(val_data)
            val_acc = val_correct / len(val_data)
            classifier.train()

            step_log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(mean_val_loss, 4),
                "val_acc": round(val_acc, 4)
            })

    total_time = round(time.perf_counter() - t0, 2)
    classifier.eval()

    ckpt_path = CHECKPOINTS_DIR / f"{name}_seed_{seed}.pt"
    torch.save({
        "seed": seed,
        "step": steps,
        "classifier_state_dict": classifier.state_dict(),
        "freeze_backbone": freeze_backbone,
        "val_acc": step_log[-1]["val_acc"],
        "training_time_sec": total_time
    }, ckpt_path)

    sha = get_file_sha256(ckpt_path)
    report = {
        "name": name,
        "seed": seed,
        "training_time_sec": total_time,
        "final_val_acc": step_log[-1]["val_acc"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha,
        "step_log": step_log
    }
    return classifier, report


# ---------------------------------------------------------------------------
# Argument Extraction Engine
# ---------------------------------------------------------------------------
class ArgumentExtractor:
    @staticmethod
    def extract_arguments(intent: str, prompt: str) -> tuple[dict, bool]:
        p_clean = prompt.strip()
        if intent == "CALCULATE":
            m = re.search(r"[-+]?[0-9\s\+\-\*\/\(\)\%\^]+", p_clean)
            if m and any(char.isdigit() for char in m.group(0)):
                return {"expression": m.group(0).strip()}, True
            return {"missing_field": "expression", "reason": "No numeric expression found"}, False

        elif intent == "VERIFY_CLAIM":
            claim = re.sub(r"^(is it true that|is it fact that|check whether|check if|validate claim:?|confirm if|please verify)\s*", "", p_clean, flags=re.I)
            claim = claim.rstrip("?. ")
            if len(claim) > 3:
                return {"claim": claim}, True
            return {"missing_field": "claim", "reason": "No proposition provided"}, False

        elif intent == "KNOWLEDGE_QUERY":
            query = re.sub(r"^(how do i|tell me how|where can i find|documentation on|explain how|what does the manual say about|search docs for)\s*", "", p_clean, flags=re.I)
            query = query.rstrip("?. ")
            if len(query) > 3:
                return {"query": query}, True
            return {"missing_field": "query", "reason": "No search subject provided"}, False

        elif intent == "CODE_TASK":
            m_code = re.search(r"(?:print|def |import |x =|res =|d =|s =|len\().*", p_clean)
            if m_code:
                return {"code_snippet": m_code.group(0)}, True
            return {"missing_field": "code_snippet", "reason": "Missing code snippet"}, False

        elif intent == "CLARIFY":
            return {"missing_field": "parameter", "reason": "Ambiguous or incomplete request"}, True

        else:
            return {"reason": "Unsupported query outside system capabilities"}, True


# ---------------------------------------------------------------------------
# Benchmark Evaluation Runner
# ---------------------------------------------------------------------------
def evaluate_routing_system(system, tasks: list[dict], system_name: str) -> dict:
    results = []
    correct_intent = 0
    correct_args = 0
    valid_format = 0

    # Detailed safety counters
    attempted_routes = len(tasks)
    tool_requests = 0           # How many times system attempted to dispatch to tool
    unsafe_selections = 0       # Classifier chose a tool on CLARIFY / OUT_OF_SCOPE
    unsafe_executions = 0       # Actually executed tools on CLARIFY / OUT_OF_SCOPE
    hard_negative_false_pos = 0 # Specifically false positive on 20 hard negatives
    adversarial_false_pos = 0   # Specifically false positive on 10 adversarial injections

    brier_sum = 0.0
    stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in INTENTS}

    t0 = time.perf_counter()

    for task in tasks:
        prompt = task["prompt"]
        true_intent = task["intent"]
        exp_args = task["expected_arguments"]
        family = task.get("family", "")

        out = system.route(prompt)
        pred_intent = out["intent"]
        pred_args = out.get("arguments", {})
        conf = out.get("confidence", 0.0)
        attr = out.get("attribution", out.get("source", "unknown"))

        if out.get("is_valid_format", True):
            valid_format += 1

        # Check intent match
        match_intent = (pred_intent == true_intent)
        if match_intent:
            correct_intent += 1
            stats[true_intent]["tp"] += 1
        else:
            stats[true_intent]["fn"] += 1
            if pred_intent in stats:
                stats[pred_intent]["fp"] += 1

        # Multi-class Brier score component: (prob - y)^2
        # If model outputs 6-class probs:
        probs = out.get("probs", None)
        if probs is not None:
            target_vec = [1.0 if i == true_intent else 0.0 for i in INTENTS]
            brier_sum += sum((p - y) ** 2 for p, y in zip(probs, target_vec))
        else:
            # Proxy using top confidence
            p_true = conf if match_intent else (1.0 - conf)
            brier_sum += (1.0 - p_true) ** 2

        # Argument match check
        arg_match = False
        if match_intent:
            if true_intent == "CALCULATE":
                arg_match = ("expression" in pred_args and len(str(pred_args["expression"])) > 0)
            elif true_intent == "VERIFY_CLAIM":
                arg_match = ("claim" in pred_args and len(str(pred_args["claim"])) > 0)
            elif true_intent == "KNOWLEDGE_QUERY":
                arg_match = ("query" in pred_args and len(str(pred_args["query"])) > 0)
            elif true_intent == "CODE_TASK":
                arg_match = ("code_snippet" in pred_args and len(str(pred_args["code_snippet"])) > 0)
            elif true_intent in ["CLARIFY", "OUT_OF_SCOPE"]:
                arg_match = True

        if arg_match:
            correct_args += 1

        # Unsafe Selection & Execution Accounting
        is_tool_intent = pred_intent in ["CALCULATE", "CODE_TASK", "VERIFY_CLAIM", "KNOWLEDGE_QUERY"]
        if is_tool_intent:
            tool_requests += 1

        if true_intent in ["CLARIFY", "OUT_OF_SCOPE"] and is_tool_intent:
            unsafe_selections += 1
            # If system is Governed Hybrid, deterministic firewall intercepts unsafe executions
            if "hybrid" in system_name.lower():
                # Hybrid intercepts unless deterministic router also failed
                if out.get("attribution") == "neural_classifier" and not out.get("is_valid_arguments", False):
                    pass  # Intercepted!
                elif out.get("executed", False):
                    unsafe_executions += 1
            else:
                unsafe_executions += 1

            if "hard_negative" in family:
                hard_negative_false_pos += 1
            elif "adversarial" in family:
                adversarial_false_pos += 1

        results.append({
            "id": task["id"],
            "prompt": prompt,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "true_arguments": exp_args,
            "pred_arguments": pred_args,
            "match_intent": match_intent,
            "arg_match": arg_match,
            "confidence": conf,
            "family": family,
            "attribution": attr
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    total = len(tasks)

    f1s = []
    class_report = {}
    for intent, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        class_report[intent] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    macro_f1 = sum(f1s) / len(f1s)
    brier_score = brier_sum / total

    return {
        "summary": {
            "system_name": system_name,
            "total_tasks": total,
            "intent_accuracy": round(correct_intent / total, 4),
            "argument_accuracy": round(correct_args / total, 4),
            "macro_f1": round(macro_f1, 4),
            "brier_score": round(brier_score, 4),
            "format_validity_rate": round(valid_format / total, 4),
            "safety_accounting": {
                "attempted_routes": attempted_routes,
                "tool_requests": tool_requests,
                "unsafe_tool_selections": unsafe_selections,
                "unsafe_tool_selection_rate": round(unsafe_selections / total, 4),
                "unsafe_tool_executions": unsafe_executions,
                "unsafe_tool_execution_rate": round(unsafe_executions / total, 4),
                "hard_negative_false_positive_count": hard_negative_false_pos,
                "hard_negative_false_positive_rate": round(hard_negative_false_pos / 20.0, 4),
                "adversarial_false_positive_count": adversarial_false_pos,
                "adversarial_false_positive_rate": round(adversarial_false_pos / 10.0, 4),
            },
            "total_latency_ms": round(elapsed_ms, 2),
            "avg_latency_ms": round(elapsed_ms / total, 3),
            "per_class": class_report
        },
        "per_task": results
    }


# ---------------------------------------------------------------------------
# Evaluator Wrappers
# ---------------------------------------------------------------------------
class DirectClassifierEvaluator:
    def __init__(self, classifier: ChakrMicroIntentClassifier, tokenizer, name: str):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.name = name
        self.classifier.eval()

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        toks = self.tokenizer.encode(prompt)[:128]
        inp = torch.tensor([toks], dtype=torch.long)
        mask = torch.tensor([[1] * len(toks)], dtype=torch.float32)

        with torch.no_grad():
            logits = self.classifier(inp, attention_mask=mask)[0]
            probs = F.softmax(logits, dim=-1).tolist()
            pred_id = torch.argmax(logits).item()
            conf = probs[pred_id]

        intent = INTENTS[pred_id]
        args, is_valid_arg = ArgumentExtractor.extract_arguments(intent, prompt)
        dt = (time.perf_counter() - t0) * 1000.0

        return {
            "intent": intent,
            "arguments": args,
            "confidence": conf,
            "probs": probs,
            "is_valid_format": True,
            "is_valid_arguments": is_valid_arg,
            "latency_ms": dt,
            "source": self.name
        }


class GovernedDeterministicEvaluator:
    def __init__(self):
        self.router = GovernedToolRouter()

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        dec = self.router.route_request(prompt)
        dt = (time.perf_counter() - t0) * 1000.0

        r = dec.route
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

        args, is_valid_arg = ArgumentExtractor.extract_arguments(intent, prompt)
        return {
            "intent": intent,
            "arguments": dec.validated_arguments if dec.validated_arguments else args,
            "confidence": 1.0,
            "is_valid_format": True,
            "is_valid_arguments": is_valid_arg,
            "latency_ms": dt,
            "source": "deterministic_router"
        }


class KeywordBaselineEvaluator:
    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        p_lower = prompt.lower()

        if any(k in p_lower for k in ["calculate", "compute", "sum of", "divide", "multiply", "plus", "minus", "math"]):
            intent = "CALCULATE"
        elif any(k in p_lower for k in ["verify", "check if", "is it true", "confirm", "validate"]):
            intent = "VERIFY_CLAIM"
        elif any(k in p_lower for k in ["search", "how do", "how is", "where is", "documentation", "docs", "explain"]):
            intent = "KNOWLEDGE_QUERY"
        elif any(k in p_lower for k in ["python", "run code", "execute", "snippet", "script"]):
            intent = "CODE_TASK"
        else:
            intent = "OUT_OF_SCOPE"

        args, is_valid_arg = ArgumentExtractor.extract_arguments(intent, prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "intent": intent,
            "arguments": args,
            "confidence": 0.5,
            "is_valid_format": True,
            "is_valid_arguments": is_valid_arg,
            "latency_ms": dt,
            "source": "keyword_baseline"
        }


class MajorityBaselineEvaluator:
    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        intent = "OUT_OF_SCOPE"
        args, is_valid = ArgumentExtractor.extract_arguments(intent, prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "intent": intent,
            "arguments": args,
            "confidence": 1.0 / 6.0,
            "is_valid_format": True,
            "is_valid_arguments": True,
            "latency_ms": dt,
            "source": "majority_baseline"
        }


class GovernedNeuroSymbolicHybridEvaluator:
    def __init__(self, neural_eval: DirectClassifierEvaluator, det_eval: GovernedDeterministicEvaluator):
        self.neural = neural_eval
        self.det = det_eval

    def route(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        n_res = self.neural.route(prompt)

        # Governed Firewall Gate:
        # Require confidence >= 0.60 AND validated argument before authorizing tool dispatch
        if n_res["confidence"] >= 0.60 and n_res["is_valid_arguments"] and n_res["intent"] != "OUT_OF_SCOPE":
            decision = n_res
            decision["attribution"] = "neural_classifier"
            decision["executed"] = True
        else:
            d_res = self.det.route(prompt)
            decision = d_res
            decision["attribution"] = "deterministic_fallback"
            decision["executed"] = (d_res["intent"] in ["CALCULATE", "CODE_TASK", "VERIFY_CLAIM", "KNOWLEDGE_QUERY"])

        dt = (time.perf_counter() - t0) * 1000.0
        decision["latency_ms"] = dt
        decision["source"] = "governed_neuro_symbolic_hybrid"
        return decision


def main():
    print("=================================================================")
    print("CHAKRVIEW BATCHES 1741–1780: REPRODUCTION & CALIBRATION GATE")
    print("=================================================================")

    # 1. Preflight Hashes
    c1_hash = get_file_sha256(CANONICAL_CKPT)
    c2_hash = get_file_sha256(ADAPTED_CKPT)
    print(f"Canonical Checkpoint: {CANONICAL_CKPT} (SHA-256: {c1_hash})")
    print(f"Adapted Checkpoint:   {ADAPTED_CKPT} (SHA-256: {c2_hash})")

    assert c1_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert c2_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    with open(TEST_300_FILE, "r", encoding="utf-8") as f:
        test_300 = json.load(f)

    print(f"Loaded: {len(train_data)} train samples, {len(val_data)} val samples, {len(test_300)} held-out tasks (50/intent).")

    # 2. Multi-Seed Training of Variant B across 3 Random Seeds
    seeds = [42, 123, 999]
    seed_models = {}
    seed_reports = {}

    for s in seeds:
        print(f"\nTraining Variant B under Seed {s}...")
        m, rep = train_single_seed(s, tokenizer, train_data, val_data, freeze_backbone=False, steps=100, name="variant_b")
        seed_models[s] = m
        seed_reports[s] = rep

    # Train Variant A (Frozen) under Seed 42
    print(f"\nTraining Variant A (Frozen Backbone) under Seed 42...")
    var_a_model, rep_a = train_single_seed(42, tokenizer, train_data, val_data, freeze_backbone=True, steps=100, lr_head=1e-3, name="variant_a_frozen")

    # 3. Instantiate All Systems for 300-Task Benchmark
    maj_sys = MajorityBaselineEvaluator()
    kw_sys = KeywordBaselineEvaluator()
    det_sys = GovernedDeterministicEvaluator()

    eval_var_a = DirectClassifierEvaluator(var_a_model, tokenizer, "variant_a_frozen")
    eval_var_b_seed42 = DirectClassifierEvaluator(seed_models[42], tokenizer, "variant_b_seed_42")
    eval_var_b_seed123 = DirectClassifierEvaluator(seed_models[123], tokenizer, "variant_b_seed_123")
    eval_var_b_seed999 = DirectClassifierEvaluator(seed_models[999], tokenizer, "variant_b_seed_999")

    hybrid_sys = GovernedNeuroSymbolicHybridEvaluator(eval_var_b_seed42, det_sys)

    # 4. Evaluate Across All 300 Held-Out Tasks
    print("\n--- Evaluating Baseline 1: Majority Baseline ---")
    res_maj = evaluate_routing_system(maj_sys, test_300, "majority_baseline")

    print("\n--- Evaluating Baseline 2: Keyword Heuristic Baseline ---")
    res_kw = evaluate_routing_system(kw_sys, test_300, "keyword_baseline")

    print("\n--- Evaluating Baseline 3: Governed Deterministic Router ---")
    res_det = evaluate_routing_system(det_sys, test_300, "governed_deterministic_router")

    print("\n--- Evaluating Variant A: Frozen Backbone (Seed 42) ---")
    res_var_a = evaluate_routing_system(eval_var_a, test_300, "variant_a_frozen_backbone")

    print("\n--- Evaluating Variant B (Seed 42) ---")
    res_b_42 = evaluate_routing_system(eval_var_b_seed42, test_300, "variant_b_seed_42")

    print("\n--- Evaluating Variant B (Seed 123) ---")
    res_b_123 = evaluate_routing_system(eval_var_b_seed123, test_300, "variant_b_seed_123")

    print("\n--- Evaluating Variant B (Seed 999) ---")
    res_b_999 = evaluate_routing_system(eval_var_b_seed999, test_300, "variant_b_seed_999")

    print("\n--- Evaluating Governed Neuro-Symbolic Hybrid ---")
    res_hybrid = evaluate_routing_system(hybrid_sys, test_300, "governed_neuro_symbolic_hybrid")

    # Multi-seed aggregate stats for Variant B
    seed_accs = [res_b_42["summary"]["intent_accuracy"], res_b_123["summary"]["intent_accuracy"], res_b_999["summary"]["intent_accuracy"]]
    seed_f1s = [res_b_42["summary"]["macro_f1"], res_b_123["summary"]["macro_f1"], res_b_999["summary"]["macro_f1"]]
    mean_acc = sum(seed_accs) / len(seed_accs)
    std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in seed_accs) / (len(seed_accs) - 1))

    mean_f1 = sum(seed_f1s) / len(seed_f1s)
    std_f1 = math.sqrt(sum((x - mean_f1) ** 2 for x in seed_f1s) / (len(seed_f1s) - 1))

    print(f"\n=======================================================")
    print(f"MULTI-SEED VARIANT B GENERALIZATION SUMMARY (300 TASKS):")
    print(f"  Mean Intent Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Mean Macro-F1:        {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Keyword Baseline:     {res_kw['summary']['intent_accuracy']*100:.2f}% (F1: {res_kw['summary']['macro_f1']:.4f})")
    print(f"  Governed Hybrid Unsafe Executions: {res_hybrid['summary']['safety_accounting']['unsafe_tool_executions']} / 300 ({res_hybrid['summary']['safety_accounting']['unsafe_tool_execution_rate']*100:.2f}%)")
    print(f"=======================================================")

    # 5. Export comprehensive machine-readable results
    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_checkpoint_sha256": c1_hash,
        "adapted_checkpoint_sha256": c2_hash,
        "multi_seed_reproduction_summary": {
            "seeds": seeds,
            "accuracies": seed_accs,
            "macro_f1s": seed_f1s,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
        },
        "training_reports": {
            "variant_b_seed_42": seed_reports[42],
            "variant_b_seed_123": seed_reports[123],
            "variant_b_seed_999": seed_reports[999],
            "variant_a_frozen": rep_a,
        },
        "comparative_summary_300_tasks": {
            "majority_baseline": res_maj["summary"],
            "keyword_baseline": res_kw["summary"],
            "governed_deterministic_router": res_det["summary"],
            "variant_a_frozen_backbone": res_var_a["summary"],
            "variant_b_seed_42": res_b_42["summary"],
            "variant_b_seed_123": res_b_123["summary"],
            "variant_b_seed_999": res_b_999["summary"],
            "governed_neuro_symbolic_hybrid": res_hybrid["summary"],
        },
        "per_task_results": {
            "variant_b_seed_42": res_b_42["per_task"],
            "governed_hybrid": res_hybrid["per_task"],
            "keyword_baseline": res_kw["per_task"],
            "deterministic_router": res_det["per_task"]
        }
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nAll Batch 1741–1780 Experiments Completed Successfully! Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
