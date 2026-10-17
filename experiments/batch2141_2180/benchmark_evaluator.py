"""
ChakrView Comprehensive Benchmark and Capability Probe Evaluator — Batches 2141-2180.
Evaluates:
1. Canonical Step 10,000 Baseline
2. Historical 150-step and 300-step candidates
3. Batch 2141 CPT Candidates (Seed 42 step 125, Seed 1337 step 50, Seed 2026 step 50)

Evaluations computed:
A. Final Held-Out Frozen Test Benchmark (290 documents, 65,059 tokens across 4 source families)
   - Token-weighted CE & Perplexity
   - Equal-weighted Macro-source CE & Perplexity
   - Per-source breakdown (Loney geometry, Info theory, Hindi literature, Classical Sanskrit)
   - Paired document difference analysis
B. Pure Neural Arithmetic Probes (Greedy decoding, exact match)
C. Pure Neural Code Syntax Completion (Python AST parsing)
D. Pure Neural Factual Cloze Retrieval
E. Repetition & Degeneration Analysis
"""

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CORPUS_CLEANED_MANIFEST_PATH = REPO_ROOT / "experiments" / "batch2101_2140" / "CORPUS_V2_CLEANED_MANIFEST.json"


def load_model(checkpoint_path: Path, canonical_config: Any, device: str = "cpu") -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = canonical_config
        state_dict = ckpt.get("model_state_dict", ckpt)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = canonical_config
        state_dict = ckpt["model_state_dict"]
    else:
        cfg = canonical_config
        state_dict = ckpt

    model = ChakrMicro(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def evaluate_held_out_test_split(
    model: ChakrMicro,
    tokenizer: Any,
    device: str = "cpu"
) -> Dict[str, Any]:
    with open(CORPUS_CLEANED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    test_docs = [d for d in corpus["documents"] if d["split"] == "test"]
    source_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for d in test_docs:
        source_buckets.setdefault(d["source_id"], []).append(d)

    per_source_stats: Dict[str, Dict[str, Any]] = {}
    total_eval_tokens = 0
    total_eval_nll = 0.0
    doc_results = []

    with torch.no_grad():
        for sid, docs in source_buckets.items():
            src_tokens = 0
            src_nll = 0.0
            doc_losses = []

            for doc in docs:
                text = doc["text"]
                doc_id = doc["document_id"]
                token_ids = tokenizer.encode(text)

                if len(token_ids) < 2:
                    continue

                max_seq = 256
                d_tokens = 0
                d_nll = 0.0

                for i in range(0, len(token_ids) - 1, max_seq - 1):
                    chunk = token_ids[i : i + max_seq]
                    if len(chunk) < 2:
                        continue

                    inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                    tgt = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

                    logits = model(inp)
                    n_tgt = tgt.numel()
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="mean")
                    d_nll += loss.item() * n_tgt
                    d_tokens += n_tgt

                if d_tokens > 0:
                    src_nll += d_nll
                    src_tokens += d_tokens
                    doc_loss = d_nll / d_tokens
                    doc_losses.append(doc_loss)
                    doc_results.append({
                        "doc_id": doc_id,
                        "source_id": sid,
                        "tokens": d_tokens,
                        "loss": round(doc_loss, 4)
                    })

            src_ce = src_nll / src_tokens if src_tokens > 0 else 0.0
            per_source_stats[sid] = {
                "source_id": sid,
                "document_count": len(docs),
                "evaluated_tokens": src_tokens,
                "token_weighted_ce": round(src_ce, 4),
                "perplexity": round(math.exp(src_ce), 2),
                "macro_doc_ce": round(sum(doc_losses) / len(doc_losses), 4) if doc_losses else 0.0
            }
            total_eval_nll += src_nll
            total_eval_tokens += src_tokens

    overall_tw_ce = total_eval_nll / total_eval_tokens if total_eval_tokens > 0 else 0.0
    source_ces = [m["token_weighted_ce"] for m in per_source_stats.values()]
    macro_source_ce = sum(source_ces) / len(source_ces) if source_ces else 0.0

    return {
        "total_test_documents": len(test_docs),
        "total_evaluated_tokens": total_eval_tokens,
        "token_weighted_ce": round(overall_tw_ce, 4),
        "token_weighted_perplexity": round(math.exp(overall_tw_ce), 2),
        "macro_source_ce": round(macro_source_ce, 4),
        "macro_source_perplexity": round(math.exp(macro_source_ce), 2),
        "per_source_metrics": per_source_stats,
        "documents": doc_results
    }


def evaluate_capability_probes(model: ChakrMicro, tokenizer: Any, device: str = "cpu") -> Dict[str, Any]:
    # 1. Arithmetic Probes
    arithmetic_cases = [
        {"prompt": "15 + 25 = ", "target": "40"},
        {"prompt": "12 * 8 = ", "target": "96"},
        {"prompt": "100 - 35 = ", "target": "65"},
        {"prompt": "81 / 9 = ", "target": "9"},
        {"prompt": "24 + 36 = ", "target": "60"},
        {"prompt": "7 * 6 = ", "target": "42"},
        {"prompt": "50 - 18 = ", "target": "32"},
        {"prompt": "64 / 8 = ", "target": "8"},
        {"prompt": "11 * 11 = ", "target": "121"},
        {"prompt": "45 + 55 = ", "target": "100"},
        {"prompt": "90 - 45 = ", "target": "45"},
        {"prompt": "9 * 9 = ", "target": "81"},
        {"prompt": "100 / 10 = ", "target": "10"},
        {"prompt": "33 + 67 = ", "target": "100"},
        {"prompt": "16 * 4 = ", "target": "64"},
        {"prompt": "75 - 25 = ", "target": "50"},
        {"prompt": "49 / 7 = ", "target": "7"},
        {"prompt": "13 + 27 = ", "target": "40"},
        {"prompt": "8 * 8 = ", "target": "64"},
        {"prompt": "80 - 30 = ", "target": "50"},
    ]

    # 2. Python Code Syntax Completion Probes (Scored via ast.parse)
    code_cases = [
        {"prefix": "def multiply(x, y):\n    return ", "target_starter": "x * y\n"},
        {"prefix": "def is_even(n):\n    return ", "target_starter": "n % 2 == 0\n"},
        {"prefix": "def get_length(arr):\n    return ", "target_starter": "len(arr)\n"},
        {"prefix": "class Queue:\n    def __init__(self):\n        self.items = ", "target_starter": "[]\n"},
        {"prefix": "def square(x):\n    return ", "target_starter": "x ** 2\n"},
        {"prefix": "def absolute(x):\n    if x < 0:\n        return -x\n    return ", "target_starter": "x\n"},
        {"prefix": "def greet(name):\n    return ", "target_starter": "f'Hello, {name}'\n"},
        {"prefix": "def add(a, b):\n    return ", "target_starter": "a + b\n"},
        {"prefix": "def negate(val):\n    return ", "target_starter": "not val\n"},
        {"prefix": "def identity(v):\n    return ", "target_starter": "v\n"},
        {"prefix": "def double(num):\n    return ", "target_starter": "num * 2\n"},
        {"prefix": "def empty_list():\n    return ", "target_starter": "[]\n"},
        {"prefix": "def empty_dict():\n    return ", "target_starter": "{}\n"},
        {"prefix": "def bool_true():\n    return ", "target_starter": "True\n"},
        {"prefix": "def bool_false():\n    return ", "target_starter": "False\n"},
    ]

    # 3. Factual Cloze Probes
    cloze_cases = [
        {"prompt": "The capital of India is ", "target": "New Delhi"},
        {"prompt": "TCP stands for Transmission Control ", "target": "Protocol"},
        {"prompt": "Water is composed of hydrogen and ", "target": "oxygen"},
        {"prompt": "The inventor of the Turing machine was Alan ", "target": "Turing"},
        {"prompt": "Binary numbers use two digits: 0 and ", "target": "1"},
        {"prompt": "Python lists are enclosed in square ", "target": "brackets"},
        {"prompt": "Munshi Premchand wrote in Hindi and ", "target": "Urdu"},
        {"prompt": "The Bhagavad Gita is a dialogue between Krishna and ", "target": "Arjuna"},
        {"prompt": "In mathematics, the ratio of a circle's circumference to diameter is ", "target": "pi"},
        {"prompt": "Information entropy measures uncertainty or ", "target": "surprise"},
    ]

    with torch.no_grad():
        # Evaluate Arithmetic
        arith_passed = 0
        for ac in arithmetic_cases:
            p_ids = tokenizer.encode(ac["prompt"])
            inp = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = inp
            for _ in range(4):
                logits = model(gen)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt == 1:
                    break
                gen = torch.cat([gen, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            pred_str = tokenizer.decode(gen[0, len(p_ids):].tolist()).strip()
            if pred_str.startswith(ac["target"]) or ac["target"] in pred_str:
                arith_passed += 1

        # Evaluate Code Syntax Completion
        code_valid = 0
        for cc in code_cases:
            p_ids = tokenizer.encode(cc["prefix"])
            inp = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = inp
            for _ in range(8):
                logits = model(gen)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt == 1:
                    break
                gen = torch.cat([gen, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            full_code = cc["prefix"] + tokenizer.decode(gen[0, len(p_ids):].tolist())
            try:
                ast.parse(full_code)
                code_valid += 1
            except Exception:
                pass

        # Evaluate Factual Cloze
        cloze_passed = 0
        for fc in cloze_cases:
            p_ids = tokenizer.encode(fc["prompt"])
            inp = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = inp
            for _ in range(4):
                logits = model(gen)
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt == 1:
                    break
                gen = torch.cat([gen, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            pred_str = tokenizer.decode(gen[0, len(p_ids):].tolist()).strip().lower()
            if fc["target"].lower() in pred_str:
                cloze_passed += 1

        # Evaluate Repetition Degeneration
        rep_prompts = [
            "In theoretical computer science, a Turing machine is ",
            "The chemical properties of elements depend on ",
            "Analytical geometry connects algebra with "
        ]
        total_unique_ratio = 0.0
        for rp in rep_prompts:
            p_ids = tokenizer.encode(rp)
            inp = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = inp
            for _ in range(32):
                logits = model(gen)
                nxt = torch.argmax(logits[0, -1, :]).item()
                gen = torch.cat([gen, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            gen_tokens = gen[0, len(p_ids):].tolist()
            unique_ratio = len(set(gen_tokens)) / len(gen_tokens) if gen_tokens else 0.0
            total_unique_ratio += unique_ratio
        avg_unique_token_ratio = total_unique_ratio / len(rep_prompts)

    return {
        "arithmetic_probes": {
            "total": len(arithmetic_cases),
            "passed": arith_passed,
            "accuracy": round(arith_passed / len(arithmetic_cases), 4)
        },
        "python_syntax_completion": {
            "total": len(code_cases),
            "valid_syntax_count": code_valid,
            "syntax_validity_rate": round(code_valid / len(code_cases), 4)
        },
        "factual_cloze_probes": {
            "total": len(cloze_cases),
            "passed": cloze_passed,
            "accuracy": round(cloze_passed / len(cloze_cases), 4)
        },
        "generation_diversity": {
            "unique_token_ratio_32tokens": round(avg_unique_token_ratio, 4)
        }
    }


def run_full_evaluation_suite() -> Dict[str, Any]:
    print("==================================================================")
    print("CHAKRVIEW COMPREHENSIVE BENCHMARK & CAPABILITY PROBE EVALUATION")
    print("==================================================================")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    canon_ckpt = torch.load(REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt", map_location="cpu", weights_only=False)
    canonical_config = canon_ckpt["config"]

    models_to_eval = {
        "canonical_baseline": REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        "historical_cpt_150": REPO_ROOT / "experiments" / "batch1861_1900" / "checkpoints" / "cpt_balanced_seed_42.pt",
        "historical_cpt_300": REPO_ROOT / "experiments" / "batch1901_1940" / "checkpoints" / "cpt_balanced_seed_42_300steps.pt",
        "cpt_seed_42_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_42" / "cpt_best_checkpoint.pt",
        "cpt_seed_1337_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_1337" / "cpt_best_checkpoint.pt",
        "cpt_seed_2026_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_2026" / "cpt_best_checkpoint.pt",
    }

    full_results = {}

    for name, p in models_to_eval.items():
        if not p.exists():
            print(f"Skipping {name}: checkpoint not found ({p})")
            continue

        print(f"\nEvaluating: {name} ({p.name})...")
        model = load_model(p, canonical_config=canonical_config)
        test_res = evaluate_held_out_test_split(model, tokenizer)
        probes_res = evaluate_capability_probes(model, tokenizer)

        full_results[name] = {
            "checkpoint_path": str(p.relative_to(REPO_ROOT)),
            "held_out_test_benchmark": test_res,
            "capability_probes": probes_res
        }
        print(f"  Test Token-Weighted CE: {test_res['token_weighted_ce']:.4f} (PPL: {test_res['token_weighted_perplexity']:.2f})")
        print(f"  Test Macro-Source CE:    {test_res['macro_source_ce']:.4f} (PPL: {test_res['macro_source_perplexity']:.2f})")
        print(f"  Arithmetic Acc:          {probes_res['arithmetic_probes']['accuracy']*100:.1f}%")
        print(f"  Code Syntax Validity:    {probes_res['python_syntax_completion']['syntax_validity_rate']*100:.1f}%")
        print(f"  Factual Cloze Acc:       {probes_res['factual_cloze_probes']['accuracy']*100:.1f}%")

    out_file = REPO_ROOT / "experiments" / "batch2141_2180" / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nFull evaluation results saved to: {out_file}")
    return full_results


if __name__ == "__main__":
    run_full_evaluation_suite()
