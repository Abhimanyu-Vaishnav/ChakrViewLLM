"""
Standalone Validation Evaluator for ChakrView Batches 2101-2140.
Evaluates model checkpoints strictly against the validation split without touching test data.
Computes token-weighted CE, per-source CE, and equal-weighted macro-source CE.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def evaluate_validation_split(
    model: torch.nn.Module,
    tokenizer: Any,
    manifest_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates a model strictly on the validation partition of the specified manifest.
    Reports:
      - token_weighted_ce
      - macro_source_ce (equal-weighted average across source families)
      - per_source_metrics
      - per_source_tokens
      - excluded_documents
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Strictly filter for validation split only
    val_docs = [d for d in corpus["documents"] if d["split"] == "validation"]
    if not val_docs:
        raise ValueError(f"No validation documents found in manifest: {manifest_path}")

    model.eval()
    model.to(device)

    source_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for d in val_docs:
        sid = d["source_id"]
        source_buckets.setdefault(sid, []).append(d)

    per_source_metrics: Dict[str, Dict[str, Any]] = {}
    total_eval_tokens = 0
    total_eval_nll = 0.0
    excluded_docs: List[str] = []

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
                    excluded_docs.append(doc_id)
                    continue

                # Process in sequence chunks up to model context window (256)
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
                    doc_losses.append(d_nll / d_tokens)
                else:
                    excluded_docs.append(doc_id)

            src_ce = src_nll / src_tokens if src_tokens > 0 else 0.0
            src_ppl = torch.exp(torch.tensor(src_ce)).item() if src_ce > 0 else 0.0

            per_source_metrics[sid] = {
                "source_id": sid,
                "document_count": len(docs),
                "evaluated_tokens": src_tokens,
                "token_weighted_ce": round(src_ce, 4),
                "perplexity": round(src_ppl, 2),
                "macro_doc_ce": round(sum(doc_losses) / len(doc_losses), 4) if doc_losses else 0.0
            }

            total_eval_nll += src_nll
            total_eval_tokens += src_tokens

    # Overall token-weighted cross-entropy
    overall_tw_ce = total_eval_nll / total_eval_tokens if total_eval_tokens > 0 else 0.0

    # Equal-weighted macro-source cross-entropy
    source_ces = [m["token_weighted_ce"] for m in per_source_metrics.values()]
    macro_source_ce = sum(source_ces) / len(source_ces) if source_ces else 0.0

    return {
        "validation_manifest": str(manifest_path),
        "total_validation_docs": len(val_docs),
        "total_evaluated_tokens": total_eval_tokens,
        "token_weighted_ce": round(overall_tw_ce, 4),
        "token_weighted_perplexity": round(torch.exp(torch.tensor(overall_tw_ce)).item(), 2),
        "macro_source_ce": round(macro_source_ce, 4),
        "macro_source_perplexity": round(torch.exp(torch.tensor(macro_source_ce)).item(), 2),
        "source_count": len(per_source_metrics),
        "per_source_metrics": per_source_metrics,
        "excluded_documents": excluded_docs
    }
