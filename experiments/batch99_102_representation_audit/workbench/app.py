"""
Batch 101 Deliverable: Enhanced Interactive Token Diagnostics Workbench.

Extends the existing Workbench implementation with interactive token-level inspection:
1. Token Chip Visualization: Renders prompts and generated outputs with individual token boundaries.
2. Interactive Token Selection: Clicking on any token displays:
   - Token ID, safely decoded text representation (handling whitespace, bytes, special tokens).
   - Generation position / step.
   - Model / checkpoint identity.
   - Top-5 candidate tokens, probabilities, and log-probabilities computed during inference.
   - Indication of whether the token is at or near the first divergence point between checkpoints.
3. Explicit Labeling:
   - Clearly labels Teacher-Forced prompt scoring vs Free-Running generation scores.
   - Explicitly displays disclaimer that token probabilities are uncalibrated model scores, not guarantees of factual correctness.
4. Offline Persistence & Backward Compatibility:
   - Persists all step diagnostics (including top-5 candidates) in saved JSON reports.
   - Reopens saved reports offline without re-running inference or needing weights in memory.
   - Gracefully handles missing per-token candidate fields from older reports (showing 'Candidate details unavailable in this report version' rather than crashing or fabricating).
5. Robust Error Handling:
   - Missing/corrupt checkpoints.
   - Special tokens and partial UTF-8 byte sequences.
   - CLI commands `inspect`, `diagnose`, `list-reports`, `serve` fully preserved.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import http
import http.server
import json
import math
from pathlib import Path
import re
import socketserver
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    compute_file_sha256,
    load_tokenizer_artifacts,
)
from experiments.batch91_94_tool_calling_comparator.tools.calculator import (
    validate_and_execute_tool_call,
    validate_tool_call_payload,
)

REPORTS_DIR = Path("experiments/batch99_102_representation_audit/workbench/reports")
LEGACY_REPORTS_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/workbench/reports")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"

DEFAULT_DIAGNOSTIC_PROMPTS = [
    "<|im_start|>user\nRequest: add numbers\nOperand A: 3\nOperand B: 4\nRequired action: calculator addition<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nCalculate 8 plus 5.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nRequest: add numbers\nOperand A: 0\nOperand B: 9\nRequired action: calculator addition<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nCompute 2 + 6.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nTell me a short joke.<|im_end|>\n<|im_start|>assistant\n",
]


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Inspects a checkpoint path, calculates verified SHA-256 and parameter count."""
    p = Path(checkpoint_path)
    if not p.is_file():
        return {
            "exists": False,
            "path": str(p),
            "error": f"File does not exist: {checkpoint_path}",
            "valid": False,
        }

    sha = compute_file_sha256(str(p))
    size_mb = round(p.stat().st_size / (1024 * 1024), 2)
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        has_model = "model_state_dict" in data
        step = data.get("step", data.get("checkpoint_step", "N/A"))
        cond = data.get("condition_name", "N/A")
        if has_model:
            sd = data["model_state_dict"]
            if "embedding.weight" in sd and "lm_head.weight" in sd:
                param_count = sum(v.numel() for k, v in sd.items() if k != "lm_head.weight")
            else:
                param_count = sum(v.numel() for v in sd.values())
        else:
            param_count = "N/A"
        history = data.get("history", [])
        return {
            "exists": True,
            "path": str(p),
            "size_mb": size_mb,
            "sha256": sha,
            "step": step,
            "condition_name": cond,
            "param_count": param_count,
            "history_length": len(history),
            "valid": has_model,
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(p),
            "size_mb": size_mb,
            "sha256": sha,
            "error": f"Corrupted or invalid checkpoint: {e}",
            "valid": False,
        }


def safe_token_repr(tokenizer: Any, token_id: int) -> str:
    """Decodes token safely, falling back to hex/repr if byte is non-UTF8 in isolation."""
    try:
        s = tokenizer.decode([token_id])
        return repr(s)
    except Exception:
        return f"<tok_{token_id}>"


def compute_teacher_forced_token_scores(
    model: ChakrMicro,
    token_ids: List[int],
    tokenizer: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Computes teacher-forced next-token log-probabilities and top-K candidate tokens.
    For each position t (from 1 to len-1), evaluates P(token[t] | token[0..t-1]).
    """
    model.eval()
    if len(token_ids) < 2:
        return []

    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_tensor)[0]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

    scores = []
    for i in range(1, len(token_ids)):
        tgt_id = token_ids[i]
        lp = log_probs[i - 1, tgt_id].item()
        p = probs[i - 1, tgt_id].item()

        top_p, top_idx = torch.topk(probs[i - 1], k=top_k)
        candidates = []
        for rank in range(top_k):
            c_id = top_idx[rank].item()
            candidates.append({
                "token_id": c_id,
                "token_str": safe_token_repr(tokenizer, c_id),
                "prob": round(top_p[rank].item(), 4),
                "log_prob": round(log_probs[i - 1, c_id].item(), 4),
            })

        scores.append({
            "position": i,
            "token_id": tgt_id,
            "token_str": safe_token_repr(tokenizer, tgt_id),
            "log_prob": round(lp, 4),
            "prob": round(p, 4),
            "top_candidates": candidates,
        })
    return scores


def run_single_inference_with_scores(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 40,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Runs greedy inference, recording generated token IDs, log-probabilities, and top-5 candidates."""
    model.eval()
    toks = [0] + tokenizer.encode(prompt)
    gen = list(toks)
    eos_reached = False
    start_t = time.perf_counter()

    step_diagnostics = []

    with torch.no_grad():
        for step in range(max_tokens):
            logits = model(torch.tensor([gen]))[0, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            nxt = torch.argmax(logits).item()
            nxt_prob = probs[nxt].item()
            nxt_lp = log_probs[nxt].item()

            top_p, top_idx = torch.topk(probs, k=top_k)
            candidates = []
            for rank in range(top_k):
                c_id = top_idx[rank].item()
                candidates.append({
                    "token_id": c_id,
                    "token_str": safe_token_repr(tokenizer, c_id),
                    "prob": round(top_p[rank].item(), 4),
                    "log_prob": round(log_probs[c_id].item(), 4),
                })

            step_diagnostics.append({
                "step": step,
                "position": len(toks) + step,
                "token_id": nxt,
                "token_str": safe_token_repr(tokenizer, nxt),
                "prob": round(nxt_prob, 4),
                "log_prob": round(nxt_lp, 4),
                "top_candidates": candidates,
            })

            gen.append(nxt)
            if nxt == 1:
                eos_reached = True
                break

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    new_tokens = gen[len(toks):]
    try:
        raw_output = tokenizer.decode(new_tokens)
    except Exception:
        from chakrview.tokenizer.decoder import decode_bytes
        raw_bytes = decode_bytes(new_tokens, vocab=tokenizer.vocab)
        raw_output = raw_bytes.decode("utf-8", errors="replace")
    clean_output = raw_output.replace("<|im_end|>", "").strip()

    total_lp = sum(d["log_prob"] for d in step_diagnostics)
    mean_lp = total_lp / max(1, len(step_diagnostics))
    ppl = round(math.exp(-mean_lp), 2) if len(step_diagnostics) > 0 else 1.0

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "generated_token_ids": new_tokens,
        "token_count": len(new_tokens),
        "eos_reached": eos_reached,
        "latency_ms": latency_ms,
        "mean_log_prob": round(mean_lp, 4),
        "perplexity": ppl,
        "step_diagnostics": step_diagnostics,
    }


def compare_and_diagnose_checkpoints(
    ckpt_a_path: str,
    ckpt_b_path: str,
    prompts: Optional[List[str]] = None,
    max_tokens: int = 40,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Performs full side-by-side comparison AND token-level diagnostics between Checkpoint A and Checkpoint B.
    """
    start_time = datetime.datetime.now().isoformat()
    if prompts is None:
        prompts = DEFAULT_DIAGNOSTIC_PROMPTS

    meta_a = inspect_checkpoint(ckpt_a_path)
    meta_b = inspect_checkpoint(ckpt_b_path)

    if not meta_a.get("valid", False) or not meta_b.get("valid", False):
        err_msg = ""
        if not meta_a.get("valid", False):
            err_msg += f"Checkpoint A invalid: {meta_a.get('error', 'unknown error')}. "
        if not meta_b.get("valid", False):
            err_msg += f"Checkpoint B invalid: {meta_b.get('error', 'unknown error')}."
        return {
            "success": False,
            "error": err_msg,
            "checkpoint_a": meta_a,
            "checkpoint_b": meta_b,
        }

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=512)

    # Load Model A
    model_a = ChakrMicro(config)
    data_a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
    model_a.load_state_dict(data_a["model_state_dict"])
    model_a.eval()

    # Load Model B
    model_b = ChakrMicro(config)
    data_b = torch.load(ckpt_b_path, map_location="cpu", weights_only=False)
    model_b.load_state_dict(data_b["model_state_dict"])
    model_b.eval()

    diagnostics = []
    identical_count = 0
    divergence_steps = []

    for idx, prompt in enumerate(prompts):
        out_a = run_single_inference_with_scores(model_a, tokenizer, prompt, max_tokens=max_tokens, top_k=5)
        out_b = run_single_inference_with_scores(model_b, tokenizer, prompt, max_tokens=max_tokens, top_k=5)

        toks_a = out_a["generated_token_ids"]
        toks_b = out_b["generated_token_ids"]
        first_divergence_idx = None
        min_len = min(len(toks_a), len(toks_b))
        for step_i in range(min_len):
            if toks_a[step_i] != toks_b[step_i]:
                first_divergence_idx = step_i
                break
        if first_divergence_idx is None and len(toks_a) != len(toks_b):
            first_divergence_idx = min_len

        is_identical = (first_divergence_idx is None and len(toks_a) == len(toks_b))
        if is_identical:
            identical_count += 1
        elif first_divergence_idx is not None:
            divergence_steps.append(first_divergence_idx)

        # Mark first divergence in step diagnostics
        for step_diag in out_a["step_diagnostics"]:
            step_diag["is_first_divergence"] = (step_diag["step"] == first_divergence_idx)
        for step_diag in out_b["step_diagnostics"]:
            step_diag["is_first_divergence"] = (step_diag["step"] == first_divergence_idx)

        prompt_tokens = [0] + tokenizer.encode(prompt)
        prompt_scores_a = compute_teacher_forced_token_scores(model_a, prompt_tokens, tokenizer, top_k=5)
        prompt_scores_b = compute_teacher_forced_token_scores(model_b, prompt_tokens, tokenizer, top_k=5)

        divergence_detail = None
        if first_divergence_idx is not None and first_divergence_idx < min_len:
            divergence_detail = {
                "step": first_divergence_idx,
                "model_a_choice": {
                    "token_id": toks_a[first_divergence_idx],
                    "token_str": repr(tokenizer.decode([toks_a[first_divergence_idx]])),
                    "prob": out_a["step_diagnostics"][first_divergence_idx]["prob"],
                    "log_prob": out_a["step_diagnostics"][first_divergence_idx]["log_prob"],
                    "candidates": out_a["step_diagnostics"][first_divergence_idx]["top_candidates"],
                },
                "model_b_choice": {
                    "token_id": toks_b[first_divergence_idx],
                    "token_str": repr(tokenizer.decode([toks_b[first_divergence_idx]])),
                    "prob": out_b["step_diagnostics"][first_divergence_idx]["prob"],
                    "log_prob": out_b["step_diagnostics"][first_divergence_idx]["log_prob"],
                    "candidates": out_b["step_diagnostics"][first_divergence_idx]["top_candidates"],
                },
            }

        diff_lines = list(difflib.unified_diff(
            out_a["clean_output"].splitlines(keepends=True),
            out_b["clean_output"].splitlines(keepends=True),
            fromfile="Model_A",
            tofile="Model_B",
        ))

        diagnostics.append({
            "index": idx,
            "prompt": prompt,
            "prompt_token_ids": prompt_tokens,
            "prompt_scores_a": prompt_scores_a,
            "prompt_scores_b": prompt_scores_b,
            "outputs_identical": is_identical,
            "first_divergence_step": first_divergence_idx,
            "divergence_detail": divergence_detail,
            "checkpoint_a_output": out_a,
            "checkpoint_b_output": out_b,
            "diff_patch": "".join(diff_lines),
            "teacher_forced_prompt_eval": {
                "method": "TEACHER_FORCED_PROMPT_PREFIX",
                "mean_log_prob_a": round(sum(s["log_prob"] for s in prompt_scores_a) / max(1, len(prompt_scores_a)), 4),
                "mean_log_prob_b": round(sum(s["log_prob"] for s in prompt_scores_b) / max(1, len(prompt_scores_b)), 4),
            },
        })

    report = {
        "success": True,
        "title": "Interactive Token-Level Diagnostic & Comparison Report",
        "timestamp": start_time,
        "methodology_disclaimer": (
            "METHODOLOGY NOTICE: "
            "1. Teacher-Forced scores evaluate the model's likelihood on a fixed, pre-existing prefix. "
            "2. Free-Running scores evaluate the model's self-generated continuation. "
            "3. Probability and perplexity metrics are raw model score densities, NOT calibrated confidence or guarantees of truth."
        ),
        "settings": {
            "max_tokens": max_tokens,
            "num_prompts": len(prompts),
            "tokenizer_path": TOKENIZER_DIR,
            "interactive_token_diagnostics_version": "v1.0",
        },
        "checkpoint_a": meta_a,
        "checkpoint_b": meta_b,
        "summary": {
            "total_prompts": len(prompts),
            "identical_outputs_count": identical_count,
            "divergent_outputs_count": len(prompts) - identical_count,
            "divergence_rate_pct": round(((len(prompts) - identical_count) / len(prompts)) * 100, 2),
            "average_first_divergence_step": round(sum(divergence_steps) / max(1, len(divergence_steps)), 2) if divergence_steps else None,
            "model_a_avg_perplexity": round(sum(d["checkpoint_a_output"]["perplexity"] for d in diagnostics) / len(diagnostics), 2),
            "model_b_avg_perplexity": round(sum(d["checkpoint_b_output"]["perplexity"] for d in diagnostics) / len(diagnostics), 2),
        },
        "diagnostics": diagnostics,
    }

    if save_report:
        report_path = save_diagnostic_report(report)
        report["saved_report_path"] = str(report_path)

    return report


def save_diagnostic_report(report: Dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"diagnostic_report_{ts}.json"
    report["report_file"] = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def load_diagnostic_report(report_path: str) -> Dict[str, Any]:
    p = Path(report_path)
    if not p.is_file():
        # Fallback check in legacy dir
        p_legacy = LEGACY_REPORTS_DIR / p.name
        if p_legacy.is_file():
            p = p_legacy
        else:
            return {"error": f"Report file not found: {report_path}", "success": False}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_saved_reports() -> List[Dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    # Search both current and legacy reports directories
    dirs = [REPORTS_DIR]
    if LEGACY_REPORTS_DIR.exists():
        dirs.append(LEGACY_REPORTS_DIR)

    seen = set()
    for d in dirs:
        for f in sorted(d.glob("*.json"), reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            try:
                with open(f, "r", encoding="utf-8") as r:
                    data = json.load(r)
                    items.append({
                        "filename": f.name,
                        "path": str(f),
                        "title": data.get("title", f.name),
                        "timestamp": data.get("timestamp", "N/A"),
                        "summary": data.get("summary", {}),
                        "checkpoint_a": data.get("checkpoint_a", {}).get("path", "N/A"),
                        "checkpoint_b": data.get("checkpoint_b", {}).get("path", "N/A"),
                    })
            except Exception:
                pass
    return items


HTML_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChakrView Workbench - Interactive Token Diagnostics</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131c2e;
            --bg-card: #1c2840;
            --accent: #00d2ff;
            --accent-b: #ffaa00;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --border: #30363d;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --code-bg: #090d16;
            --token-divergent: rgba(248, 81, 73, 0.35);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-primary); padding: 24px; }
        .container { max-width: 1400px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { font-size: 24px; font-weight: 700; background: linear-gradient(90deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .badge { background: var(--bg-card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 12px; font-size: 12px; color: var(--accent); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        h2 { font-size: 16px; margin-bottom: 14px; color: var(--accent); display: flex; align-items: center; gap: 8px; }
        label { display: block; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
        input, select, textarea { width: 100%; background: var(--bg-primary); border: 1px solid var(--border); color: var(--text-primary); padding: 10px; border-radius: 6px; font-size: 14px; margin-bottom: 14px; }
        textarea { resize: vertical; min-height: 90px; font-family: monospace; font-size: 12px; }
        button { background: linear-gradient(135deg, #00d2ff, #0072ff); color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        .meta-box { background: var(--code-bg); border: 1px solid var(--border); padding: 10px; border-radius: 6px; font-size: 12px; font-family: monospace; margin-top: 10px; word-break: break-all; }
        .diagnostic-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
        .diagnostic-header { padding: 12px 16px; background: rgba(0,0,0,0.25); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 600; }
        .diagnostic-body { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 16px; }
        .output-col { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 14px; }
        .output-col h4 { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; display: flex; justify-content: space-between; }
        .token-flow { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
        .token-chip {
            display: inline-flex; align-items: center; gap: 4px;
            background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 4px; padding: 3px 6px; font-family: monospace; font-size: 12px;
            cursor: pointer; user-select: none; transition: all 0.15s ease;
        }
        .token-chip:hover { background: rgba(0, 210, 255, 0.2); border-color: var(--accent); }
        .token-chip.selected { background: var(--accent); color: #000; font-weight: bold; border-color: var(--accent); }
        .token-chip.divergent-first { border: 1.5px solid var(--danger); background: var(--token-divergent); }
        .divergence-box { margin: 0 16px 14px 16px; padding: 12px 14px; background: rgba(248, 81, 73, 0.08); border-left: 3px solid var(--danger); font-size: 12px; border-radius: 4px; }
        .method-banner { margin: 0 16px 10px 16px; padding: 8px 12px; background: rgba(0, 210, 255, 0.08); border-left: 3px solid var(--accent); font-size: 11px; border-radius: 4px; color: var(--text-secondary); }
        .diff-tag { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: bold; }
        .diff-tag.identical { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
        .diff-tag.divergent { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); }
        .summary-stats { display: flex; gap: 16px; margin-bottom: 20px; }
        .stat-card { flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }
        .stat-value { font-size: 20px; font-weight: bold; color: var(--accent); margin-top: 4px; }
        .stat-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; }

        /* Token Detail Drawer */
        #tokenInspectorCard {
            position: fixed; bottom: 20px; right: 20px; width: 440px;
            background: var(--bg-secondary); border: 1.5px solid var(--accent);
            border-radius: 10px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            display: none; z-index: 1000;
        }
        #tokenInspectorCard h3 { font-size: 14px; color: var(--accent); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .cand-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; font-family: monospace; }
        .cand-table th, .cand-table td { padding: 4px 6px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .cand-table th { color: var(--text-secondary); text-transform: uppercase; }
        .cand-table tr:hover { background: rgba(255,255,255,0.04); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>ChakrView Interactive Token Diagnostics</h1>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Per-Token Probabilities · Divergence Tracing · Offline Diagnostic Replay</div>
            </div>
            <div class="badge" id="appStatus">Ready · Interactive Diagnostics v1.0</div>
        </header>

        <div class="grid-2">
            <div class="card">
                <h2><span>🔍</span> Checkpoint A (Reference)</h2>
                <label>Checkpoint Path</label>
                <input type="text" id="pathA" value="experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt" />
                <button onclick="inspectCkpt('A')">Inspect A</button>
                <div id="metaA" class="meta-box">Click Inspect to view SHA-256 and parameter stats</div>
            </div>

            <div class="card">
                <h2><span>🔍</span> Checkpoint B (Candidate)</h2>
                <label>Checkpoint Path</label>
                <input type="text" id="pathB" value="experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt" />
                <button onclick="inspectCkpt('B')">Inspect B</button>
                <div id="metaB" class="meta-box">Click Inspect to view SHA-256 and parameter stats</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 24px;">
            <h2><span>⚡</span> Run Diagnostics & Divergence Analysis</h2>
            <label>Prompts (One per line)</label>
            <textarea id="promptList">Request: add numbers\nOperand A: 3\nOperand B: 4\nRequired action: calculator addition
Calculate 8 plus 5.
Request: add numbers\nOperand A: 0\nOperand B: 9\nRequired action: calculator addition
Compute 2 + 6.
Tell me a short joke.</textarea>
            <div style="display: flex; gap: 14px; align-items: center;">
                <label style="margin: 0;">Max Tokens: <input type="number" id="maxTokens" value="40" style="width: 80px; display: inline-block; margin: 0 0 0 8px;" /></label>
                <button onclick="executeDiagnostics()">Run Diagnostics</button>
                <button style="background: var(--bg-card); border: 1px solid var(--border);" onclick="loadSavedReport()">Reopen Saved Report (Offline)</button>
            </div>
        </div>

        <div id="resultsArea" style="display: none;">
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-label">Total Prompts</div>
                    <div class="stat-value" id="statPrompts">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Divergence Rate</div>
                    <div class="stat-value" id="statDivergence">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg First Divergence Step</div>
                    <div class="stat-value" id="statAvgDivergence">N/A</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Model A Avg Perplexity</div>
                    <div class="stat-value" id="statPplA">0.0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Model B Avg Perplexity</div>
                    <div class="stat-value" id="statPplB">0.0</div>
                </div>
            </div>

            <div id="diagnosticList"></div>
        </div>
    </div>

    <!-- Interactive Token Inspector Modal -->
    <div id="tokenInspectorCard">
        <h3>
            <span>Token Diagnostics</span>
            <span style="cursor:pointer; font-size:16px; color:var(--text-secondary);" onclick="closeInspector()">✕</span>
        </h3>
        <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">
            <span id="inspModel">Model</span> · Step <span id="inspStep">0</span> (Seq Pos: <span id="inspPos">0</span>)
        </div>
        <div style="background:var(--code-bg); padding:8px; border-radius:6px; margin-bottom:10px; font-family:monospace; font-size:12px;">
            Token ID: <strong id="inspId">0</strong> | Text: <strong id="inspText">''</strong><br>
            Prob: <span id="inspProb">0%</span> | Log-Prob: <span id="inspLogProb">0.00</span>
            <div id="inspDivergenceBadge" style="margin-top:4px; font-size:11px; font-weight:bold; color:var(--danger); display:none;">⚠️ FIRST DIVERGENCE POINT</div>
        </div>
        <div style="font-size:11px; color:var(--text-secondary); font-weight:bold; margin-bottom:4px;">TOP-5 CANDIDATES:</div>
        <div id="inspCandidatesArea"></div>
        <div style="font-size:10px; color:var(--text-secondary); margin-top:8px; font-style:italic;">
            Note: Probabilities are raw model scores, not guarantees of factual correctness.
        </div>
    </div>

    <script>
        let currentReport = null;

        async function inspectCkpt(tag) {
            const path = document.getElementById(tag === 'A' ? 'pathA' : 'pathB').value;
            const resBox = document.getElementById(tag === 'A' ? 'metaA' : 'metaB');
            resBox.innerText = "Inspecting...";
            try {
                const resp = await fetch('/api/inspect?path=' + encodeURIComponent(path));
                const data = await resp.json();
                if (data.valid) {
                    resBox.innerHTML = `<strong>SHA-256:</strong> ${data.sha256}<br><strong>Params:</strong> ${data.param_count.toLocaleString()}<br><strong>Size:</strong> ${data.size_mb} MB | Step: ${data.step}`;
                } else {
                    resBox.innerHTML = `<span style="color:var(--danger)">Error: ${data.error || 'Invalid file'}</span>`;
                }
            } catch (e) {
                resBox.innerHTML = `<span style="color:var(--danger)">Request failed: ${e}</span>`;
            }
        }

        async function executeDiagnostics() {
            const pathA = document.getElementById('pathA').value;
            const pathB = document.getElementById('pathB').value;
            const rawText = document.getElementById('promptList').value;
            const lines = rawText.split('\\n\\n').map(p => p.trim()).filter(p => p.length > 0);
            const maxTokens = parseInt(document.getElementById('maxTokens').value) || 40;

            const appStatus = document.getElementById('appStatus');
            appStatus.innerText = "Computing Inference & Token Probabilities...";
            appStatus.style.color = "var(--warning)";

            try {
                const formattedPrompts = lines.map(p => {
                    const cleanP = p.replace(/\\\\n/g, '\\n');
                    return `<|im_start|>user\\n${cleanP}<|im_end|>\\n<|im_start|>assistant\\n`;
                });
                const resp = await fetch('/api/diagnose', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        ckpt_a: pathA,
                        ckpt_b: pathB,
                        prompts: formattedPrompts,
                        max_tokens: maxTokens
                    })
                });
                const report = await resp.json();
                currentReport = report;
                renderReport(report);
                appStatus.innerText = "Diagnostics Complete";
                appStatus.style.color = "var(--success)";
            } catch (e) {
                alert("Diagnostic error: " + e);
                appStatus.innerText = "Failed";
                appStatus.style.color = "var(--danger)";
            }
        }

        function renderReport(report) {
            if (!report.success) {
                alert("Diagnostics failed: " + report.error);
                return;
            }
            currentReport = report;
            document.getElementById('resultsArea').style.display = 'block';
            document.getElementById('statPrompts').innerText = report.summary.total_prompts;
            document.getElementById('statDivergence').innerText = report.summary.divergence_rate_pct + "%";
            document.getElementById('statAvgDivergence').innerText = report.summary.average_first_divergence_step !== null ? `Step ${report.summary.average_first_divergence_step}` : 'Identical';
            document.getElementById('statPplA').innerText = report.summary.model_a_avg_perplexity;
            document.getElementById('statPplB').innerText = report.summary.model_b_avg_perplexity;

            const list = document.getElementById('diagnosticList');
            list.innerHTML = "";

            report.diagnostics.forEach((item, promptIdx) => {
                const isIdentical = item.outputs_identical;
                const divDetail = item.divergence_detail;
                let divHtml = '';
                if (divDetail) {
                    const cA = divDetail.model_a_choice;
                    const cB = divDetail.model_b_choice;
                    divHtml = `
                        <div class="divergence-box">
                            <strong>[FIRST TOKEN DIVERGENCE AT GENERATION STEP ${divDetail.step}]</strong><br>
                            <strong>Model A:</strong> <code>${escapeHtml(cA.token_str)}</code> (prob: ${(cA.prob*100).toFixed(1)}%, log_prob: ${cA.log_prob})<br>
                            <strong>Model B:</strong> <code>${escapeHtml(cB.token_str)}</code> (prob: ${(cB.prob*100).toFixed(1)}%, log_prob: ${cB.log_prob})
                        </div>
                    `;
                }

                const node = document.createElement('div');
                node.className = 'diagnostic-item';

                // Render token chips for Model A and Model B
                const tokensHtmlA = renderTokenChips(item.checkpoint_a_output.step_diagnostics, promptIdx, 'A');
                const tokensHtmlB = renderTokenChips(item.checkpoint_b_output.step_diagnostics, promptIdx, 'B');

                node.innerHTML = `
                    <div class="diagnostic-header">
                        <div>Prompt ${promptIdx + 1}: <code>${escapeHtml(item.prompt.replace('<|im_start|>user\\n', '').replace('<|im_end|>\\n<|im_start|>assistant\\n', ''))}</code></div>
                        <div class="diff-tag ${isIdentical ? 'identical' : 'divergent'}">${isIdentical ? 'IDENTICAL' : 'DIVERGENT'}</div>
                    </div>
                    <div class="method-banner">
                        <strong>TEACHER-FORCED PROMPT LOG-PROB:</strong> Model A = ${item.teacher_forced_prompt_eval.mean_log_prob_a} | Model B = ${item.teacher_forced_prompt_eval.mean_log_prob_b}
                        <span style="float:right; opacity:0.8;">(Evaluated on identical fixed prefix)</span>
                    </div>
                    ${divHtml}
                    <div class="diagnostic-body">
                        <div class="output-col">
                            <h4>
                                <span>Model A (Free-Running)</span>
                                <span>${item.checkpoint_a_output.latency_ms}ms · ppl: ${item.checkpoint_a_output.perplexity}</span>
                            </h4>
                            <div class="token-flow">${tokensHtmlA}</div>
                        </div>
                        <div class="output-col">
                            <h4>
                                <span>Model B (Free-Running)</span>
                                <span>${item.checkpoint_b_output.latency_ms}ms · ppl: ${item.checkpoint_b_output.perplexity}</span>
                            </h4>
                            <div class="token-flow">${tokensHtmlB}</div>
                        </div>
                    </div>
                `;
                list.appendChild(node);
            });
        }

        function renderTokenChips(stepDiagnostics, promptIdx, modelTag) {
            if (!stepDiagnostics || stepDiagnostics.length === 0) {
                return '<span style="color:var(--text-secondary); font-size:12px;">&lt;No tokens generated&gt;</span>';
            }
            return stepDiagnostics.map((st, stepIdx) => {
                const divClass = st.is_first_divergence ? 'divergent-first' : '';
                return `
                    <span class="token-chip ${divClass}"
                          id="chip_${promptIdx}_${modelTag}_${stepIdx}"
                          onclick="inspectToken(${promptIdx}, '${modelTag}', ${stepIdx})"
                          title="Step ${st.step}: Token ${st.token_id} | Prob: ${(st.prob*100).toFixed(1)}%">
                        ${escapeHtml(st.token_str.replace(/^'|'$/g, ''))}
                    </span>
                `;
            }).join('');
        }

        function inspectToken(promptIdx, modelTag, stepIdx) {
            if (!currentReport) return;
            const item = currentReport.diagnostics[promptIdx];
            const out = modelTag === 'A' ? item.checkpoint_a_output : item.checkpoint_b_output;
            const diag = out.step_diagnostics[stepIdx];
            if (!diag) return;

            document.querySelectorAll('.token-chip.selected').forEach(c => c.classList.remove('selected'));
            const selectedChip = document.getElementById(`chip_${promptIdx}_${modelTag}_${stepIdx}`);
            if (selectedChip) selectedChip.classList.add('selected');

            document.getElementById('inspModel').innerText = `Model ${modelTag} (${currentReport['checkpoint_' + modelTag.toLowerCase()].condition_name || 'Checkpoint'})`;
            document.getElementById('inspStep').innerText = diag.step;
            document.getElementById('inspPos').innerText = diag.position || (item.prompt_token_ids ? item.prompt_token_ids.length + diag.step : diag.step);
            document.getElementById('inspId').innerText = diag.token_id;
            document.getElementById('inspText').innerText = diag.token_str;
            document.getElementById('inspProb').innerText = (diag.prob * 100).toFixed(2) + "%";
            document.getElementById('inspLogProb').innerText = diag.log_prob.toFixed(4);

            const divBadge = document.getElementById('inspDivergenceBadge');
            divBadge.style.display = diag.is_first_divergence ? 'block' : 'none';

            const candsArea = document.getElementById('inspCandidatesArea');
            if (diag.top_candidates && diag.top_candidates.length > 0) {
                let tableHtml = `
                    <table class="cand-table">
                        <thead>
                            <tr><th>Rank</th><th>Token</th><th>ID</th><th>Prob</th><th>Log-P</th></tr>
                        </thead>
                        <tbody>
                `;
                diag.top_candidates.forEach((c, r) => {
                    tableHtml += `
                        <tr>
                            <td>#${r + 1}</td>
                            <td><strong>${escapeHtml(c.token_str)}</strong></td>
                            <td>${c.token_id}</td>
                            <td>${(c.prob * 100).toFixed(2)}%</td>
                            <td>${c.log_prob !== undefined ? c.log_prob.toFixed(3) : 'N/A'}</td>
                        </tr>
                    `;
                });
                tableHtml += '</tbody></table>';
                candsArea.innerHTML = tableHtml;
            } else {
                candsArea.innerHTML = '<div style="color:var(--text-secondary); font-size:11px; padding:6px 0;">Candidate details unavailable in this report version.</div>';
            }

            document.getElementById('tokenInspectorCard').style.display = 'block';
        }

        function closeInspector() {
            document.getElementById('tokenInspectorCard').style.display = 'none';
            document.querySelectorAll('.token-chip.selected').forEach(c => c.classList.remove('selected'));
        }

        async function loadSavedReport() {
            const filename = prompt("Enter report filename or leave blank to open latest saved report:");
            try {
                const resp = await fetch('/api/reports' + (filename ? '?file=' + encodeURIComponent(filename) : ''));
                const data = await resp.json();
                if (data.report) {
                    renderReport(data.report);
                    document.getElementById('appStatus').innerText = "Report Loaded: " + (filename || data.report.report_file || 'Latest');
                } else if (data.reports) {
                    alert("Available reports:\\n" + data.reports.map(r => r.filename).join('\\n'));
                }
            } catch (e) {
                alert("Failed to load report offline: " + e);
            }
        }

        function escapeHtml(text) {
            return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
    </script>
</body>
</html>
"""


class WorkbenchHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_INDEX.encode("utf-8"))
        elif self.path.startswith("/api/inspect"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            ckpt_path = qs.get("path", [""])[0]
            data = inspect_checkpoint(ckpt_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif self.path.startswith("/api/reports"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            fname = qs.get("file", [""])[0]
            if fname:
                rep = load_diagnostic_report(str(REPORTS_DIR / fname))
                resp = {"report": rep}
            else:
                reps = list_saved_reports()
                latest = load_diagnostic_report(reps[0]["path"]) if reps else None
                resp = {"reports": reps, "report": latest}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/diagnose":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            req = json.loads(body)
            ckpt_a = req.get("ckpt_a", "")
            ckpt_b = req.get("ckpt_b", "")
            prompts = req.get("prompts", DEFAULT_DIAGNOSTIC_PROMPTS)
            max_tokens = req.get("max_tokens", 40)
            res = compare_and_diagnose_checkpoints(ckpt_a, ckpt_b, prompts=prompts, max_tokens=max_tokens, save_report=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_cli():
    parser = argparse.ArgumentParser(description="ChakrView Interactive Token Diagnostics Workbench (Batches 99-102)")
    subparsers = parser.add_subparsers(dest="command")

    inspect_cmd = subparsers.add_parser("inspect", help="Inspect a checkpoint")
    inspect_cmd.add_argument("--ckpt", required=True, help="Path to checkpoint")

    diag_cmd = subparsers.add_parser("diagnose", help="Run token-level diagnostics between two checkpoints")
    diag_cmd.add_argument("--ckpt-a", required=True, help="Path to Checkpoint A")
    diag_cmd.add_argument("--ckpt-b", required=True, help="Path to Checkpoint B")
    diag_cmd.add_argument("--max-tokens", type=int, default=40, help="Max tokens per prompt")
    diag_cmd.add_argument("--prompts", default=None, help="Optional semicolon-separated prompts")

    subparsers.add_parser("list-reports", help="List saved diagnostic reports")

    serve_cmd = subparsers.add_parser("serve", help="Launch Local Workbench Web Server")
    serve_cmd.add_argument("--port", type=int, default=8501, help="Port to serve on")

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_checkpoint(args.ckpt)
        print(json.dumps(info, indent=2))
    elif args.command == "diagnose":
        prompts = [p.strip() for p in args.prompts.split(";")] if args.prompts else None
        res = compare_and_diagnose_checkpoints(args.ckpt_a, args.ckpt_b, prompts=prompts, max_tokens=args.max_tokens)
        print(f"\n=== INTERACTIVE TOKEN DIAGNOSTIC REPORT ===")
        print(f"Checkpoint A: {res.get('checkpoint_a', {}).get('path')}")
        print(f"Checkpoint B: {res.get('checkpoint_b', {}).get('path')}")
        if not res.get("success"):
            print(f"FAILED: {res.get('error')}")
            sys.exit(1)
        summary = res["summary"]
        print(f"Prompts: {summary['total_prompts']} | Identical: {summary['identical_outputs_count']} | Divergent: {summary['divergent_outputs_count']} ({summary['divergence_rate_pct']}%)")
        print(f"Avg First Divergence Step: {summary['average_first_divergence_step']}")
        print(f"Model A Avg PPL: {summary['model_a_avg_perplexity']} | Model B Avg PPL: {summary['model_b_avg_perplexity']}")
        print(f"Saved Report: {res.get('saved_report_path')}")
        for c in res["diagnostics"]:
            print(f"\n--- Prompt: {repr(c['prompt'][:50])}... ---")
            print(f"  [CKPT A]: {repr(c['checkpoint_a_output']['clean_output'])}")
            print(f"  [CKPT B]: {repr(c['checkpoint_b_output']['clean_output'])}")
            print(f"  [FIRST DIVERGENCE]: Step {c['first_divergence_step']}")
            if c.get("divergence_detail"):
                det = c["divergence_detail"]
                print(f"    Choice A: {det['model_a_choice']['token_str']} (prob: {det['model_a_choice']['prob']})")
                print(f"    Choice B: {det['model_b_choice']['token_str']} (prob: {det['model_b_choice']['prob']})")
    elif args.command == "list-reports":
        reps = list_saved_reports()
        print(f"Found {len(reps)} reports in {REPORTS_DIR}:")
        for r in reps:
            print(f"- {r['filename']}: {r['title']} ({r['timestamp']})")
    elif args.command == "serve":
        print(f"Starting ChakrView Workbench Diagnostic Server on http://localhost:{args.port}...")
        server = socketserver.TCPServer(("", args.port), WorkbenchHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
    else:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
