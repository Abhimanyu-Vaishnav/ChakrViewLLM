"""
Reproducible evaluation script comparing Baseline and Candidate checkpoints on the
Independent Multi-Domain Evaluation Corpus (Batch 1421–1460).
Location: experiments/batch1421_1460/run_independent_evaluation.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1421_1460/data/corpus")
MANIFEST_PATH = Path("experiments/batch1421_1460/data/independent_corpus_manifest.json")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
CANDIDATE_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
OUT_DIR = Path("experiments/batch1421_1460/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_on_corpus(model: ChakrMicro, tokenizer, documents: list[dict]) -> dict:
    model.eval()
    total_nll = 0.0
    total_targets = 0
    top1_c = 0
    top5_c = 0
    top10_c = 0
    per_doc = []
    cat_summary = {}

    t0 = time.perf_counter()
    with torch.no_grad():
        for doc in documents:
            filepath = CORPUS_DIR / doc["filename"]
            txt = filepath.read_text(encoding="utf-8")
            tokens = tokenizer.encode(txt)
            if len(tokens) < 2:
                continue

            targets = tokens[1:]
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            target_tensor = torch.tensor(targets, dtype=torch.long)

            logits = model(input_ids)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
            doc_nll = -target_lp.sum().item()

            total_nll += doc_nll
            total_targets += len(targets)

            _, top10 = torch.topk(logits, k=10, dim=-1)
            for i, target_id in enumerate(targets):
                top10_list = top10[i].tolist()
                if target_id == top10_list[0]:
                    top1_c += 1
                if target_id in top10_list[:5]:
                    top5_c += 1
                if target_id in top10_list:
                    top10_c += 1

            cat = doc["category"]
            if cat not in cat_summary:
                cat_summary[cat] = {"docs": 0, "targets": 0, "nll": 0.0, "top1": 0, "top10": 0}
            cat_summary[cat]["docs"] += 1
            cat_summary[cat]["targets"] += len(targets)
            cat_summary[cat]["nll"] += doc_nll

            doc_loss = doc_nll / len(targets)
            per_doc.append({
                "filename": doc["filename"],
                "category": cat,
                "targets": len(targets),
                "loss": round(doc_loss, 4),
                "ppl": round(math.exp(doc_loss), 2)
            })

    duration = time.perf_counter() - t0
    mean_loss = total_nll / total_targets
    mean_ppl = math.exp(mean_loss)

    # Category aggregations
    cat_results = {}
    for cat, data in sorted(cat_summary.items()):
        c_loss = data["nll"] / data["targets"]
        cat_results[cat] = {
            "documents": data["docs"],
            "targets": data["targets"],
            "cross_entropy": round(c_loss, 4),
            "perplexity": round(math.exp(c_loss), 2)
        }

    return {
        "targets_evaluated": total_targets,
        "duration_seconds": round(duration, 3),
        "throughput_tokens_per_sec": round(total_targets / duration, 1),
        "cross_entropy_loss": round(mean_loss, 4),
        "perplexity": round(mean_ppl, 2),
        "top_1_accuracy": round(top1_c / total_targets, 4),
        "top_5_accuracy": round(top5_c / total_targets, 4),
        "top_10_accuracy": round(top10_c / total_targets, 4),
        "category_breakdown": cat_results,
        "per_document": per_doc
    }


def main():
    print("Loading tokenizer...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    print(f"Loaded independent manifest: {len(documents)} documents across {len(manifest['categories'])} categories.")

    # 1. Evaluate Canonical Baseline
    print("\n--- Evaluating Canonical Baseline (Step 10,000) ---")
    base_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    base_model = ChakrMicro(base_ckpt["config"])
    base_model.load_state_dict(base_ckpt["model_state_dict"], strict=True)
    base_metrics = evaluate_model_on_corpus(base_model, tokenizer, documents)
    print(f"Baseline Loss: {base_metrics['cross_entropy_loss']} | PPL: {base_metrics['perplexity']} | Top-1: {base_metrics['top_1_accuracy']*100:.2f}% | Top-10: {base_metrics['top_10_accuracy']*100:.2f}%")

    # 2. Evaluate Adapted Candidate
    print("\n--- Evaluating Domain-Adapted Candidate (Step 10,080) ---")
    cand_ckpt = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)
    cand_model = ChakrMicro(cand_ckpt["config"])
    cand_model.load_state_dict(cand_ckpt["model_state_dict"], strict=True)
    cand_metrics = evaluate_model_on_corpus(cand_model, tokenizer, documents)
    print(f"Candidate Loss: {cand_metrics['cross_entropy_loss']} | PPL: {cand_metrics['perplexity']} | Top-1: {cand_metrics['top_1_accuracy']*100:.2f}% | Top-10: {cand_metrics['top_10_accuracy']*100:.2f}%")

    # 3. Calculate Deltas
    loss_delta = cand_metrics["cross_entropy_loss"] - base_metrics["cross_entropy_loss"]
    ppl_delta = cand_metrics["perplexity"] - base_metrics["perplexity"]
    top1_delta = cand_metrics["top_1_accuracy"] - base_metrics["top_1_accuracy"]
    top10_delta = cand_metrics["top_10_accuracy"] - base_metrics["top_10_accuracy"]

    # 4. Save Comprehensive Machine-Readable Output
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_manifest_sha256": hashlib_sha256(MANIFEST_PATH),
        "total_documents": len(documents),
        "total_targets_evaluated": base_metrics["targets_evaluated"],
        "baseline_canonical": {
            "checkpoint_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
            "metrics": base_metrics
        },
        "candidate_adapted": {
            "checkpoint_sha256": "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139",
            "metrics": cand_metrics
        },
        "deltas": {
            "loss_delta": round(loss_delta, 4),
            "perplexity_delta": round(ppl_delta, 2),
            "top_1_accuracy_delta": round(top1_delta, 4),
            "top_10_accuracy_delta": round(top10_delta, 4),
            "perplexity_relative_reduction_percent": round((base_metrics["perplexity"] - cand_metrics["perplexity"]) / base_metrics["perplexity"] * 100, 2)
        },
        "per_document_comparison": [
            {
                "filename": b["filename"],
                "category": b["category"],
                "targets": b["targets"],
                "baseline_loss": b["loss"],
                "candidate_loss": c["loss"],
                "loss_delta": round(c["loss"] - b["loss"], 4),
                "improved": (c["loss"] < b["loss"])
            }
            for b, c in zip(base_metrics["per_document"], cand_metrics["per_document"])
        ]
    }

    out_file = OUT_DIR / "INDEPENDENT_EVALUATION_RESULTS.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable independent evaluation results to {out_file}")

    print("\n=========================================================================================")
    print("INDEPENDENT GENERALIZATION BENCHMARK (30 DOCS, 7,741 UNSEEN TARGETS)")
    print("=========================================================================================")
    print(f"{'Category':<32} | {'Baseline PPL':<15} | {'Candidate PPL':<15} | {'PPL Delta':<12}")
    print("-" * 85)
    for cat in manifest["categories"]:
        b_ppl = base_metrics["category_breakdown"][cat]["perplexity"]
        c_ppl = cand_metrics["category_breakdown"][cat]["perplexity"]
        print(f"{cat:<32} | {b_ppl:<15.2f} | {c_ppl:<15.2f} | {c_ppl - b_ppl:<+12.2f}")
    print("-" * 85)
    print(f"{'OVERALL AGGREGATE':<32} | {base_metrics['perplexity']:<15.2f} | {cand_metrics['perplexity']:<15.2f} | {ppl_delta:<+12.2f}")
    print(f"{'Top-1 Accuracy':<32} | {base_metrics['top_1_accuracy']*100:<14.2f}% | {cand_metrics['top_1_accuracy']*100:<14.2f}% | {top1_delta*100:<+11.2f}%")
    print(f"{'Top-10 Accuracy':<32} | {base_metrics['top_10_accuracy']*100:<14.2f}% | {cand_metrics['top_10_accuracy']*100:<14.2f}% | {top10_delta*100:<+11.2f}%")
    print("=========================================================================================\n")


def hashlib_sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
