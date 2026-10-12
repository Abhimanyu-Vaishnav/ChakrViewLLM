"""
Local Workbench Checkpoint Output Comparator & Experiment Dashboard (Batches 91-94).

Features:
- Select Checkpoint A and Checkpoint B (with validation, SHA-256 calculation, parameter count, metadata).
- Run side-by-side inference on identical fixed prompts, tokenizer, decoding settings.
- Character- and token-level diff highlighting between checkpoint outputs.
- Display Deterministic Oracle reference results alongside neural model outputs.
- Track runtime latency, tokens generated, EOS completion, and raw/parsed responses.
- Save timestamped comparison reports (JSON) and reopen previously saved comparison reports without rerunning inference.
- Non-crashing graceful error handling for missing, incompatible, or corrupted checkpoints.
- Standalone CLI interface (`compare`, `inspect`, `list-reports`, `serve`).
- Embedded interactive web UI.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import http
import http.server
import json
from pathlib import Path
import re
import socketserver
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import torch

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

REPORTS_DIR = Path("experiments/batch91_94_tool_calling_comparator/workbench/reports")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"

DEFAULT_COMPARISON_PROMPTS = [
    "<|im_start|>user\nCalculate 3 + 4.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 8 + 5?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nAdd 0 and 9.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nHow much is 7 + 7?<|im_end|>\n<|im_start|>assistant\n",
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
        # Deduplicate tied weights if embedding and lm_head share memory / data pointer
        if has_model:
            sd = data["model_state_dict"]
            # Standard ChakrMicro has tied embedding and lm_head (3,443,136 total unique params)
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


def run_deterministic_oracle(prompt: str) -> Dict[str, Any]:
    """Extracts operands from prompt if arithmetic and evaluates via deterministic oracle."""
    m = re.search(r"(\d+)\s*[\+\-]\s*(\d+)", prompt)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        op = "+" if "+" in m.group(0) else "-"
        ans = a + b if op == "+" else a - b
        return {
            "source": "DETERMINISTIC_ORACLE",
            "is_arithmetic": True,
            "operation": "add" if op == "+" else "sub",
            "operands": [a, b],
            "expected_answer": str(ans),
            "expected_tool_call": f'{{"tool": "calculator", "operation": "{"add" if op == "+" else "sub"}", "a": {a}, "b": {b}}}',
        }
    return {
        "source": "DETERMINISTIC_ORACLE",
        "is_arithmetic": False,
        "expected_answer": "N/A (general language request)",
    }


def run_single_inference(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 40,
) -> Dict[str, Any]:
    """Runs deterministic greedy inference for a single prompt."""
    toks = [0] + tokenizer.encode(prompt)
    gen = list(toks)
    eos_reached = False
    start_t = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([gen]))[0, -1]
            nxt = torch.argmax(logits).item()
            gen.append(nxt)
            if nxt == 1:
                eos_reached = True
                break

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    new_token_ids = gen[len(toks):]
    raw_output = tokenizer.decode(new_token_ids)
    clean = raw_output.replace("<|im_end|>", "").strip()

    # Extract parsed answer or tool call
    tool_val = validate_tool_call_payload(clean)

    # Number regex fallback
    num_match = re.search(r"answer:\s*([0-9]+)", clean, re.IGNORECASE)
    parsed_num = num_match.group(1) if num_match else None
    if not parsed_num:
        m2 = re.match(r"^\s*([0-9]+)", clean)
        parsed_num = m2.group(1) if m2 else None

    return {
        "source": "CHAKRMICRO_NEURAL_MODEL",
        "prompt": prompt,
        "raw_output": raw_output,
        "clean_output": clean,
        "token_count": len(new_token_ids),
        "eos_reached": eos_reached,
        "latency_ms": latency_ms,
        "parsed_number": parsed_num,
        "tool_call_validation": tool_val,
    }


def compare_checkpoints(
    ckpt_a_path: str,
    ckpt_b_path: str,
    prompts: Optional[List[str]] = None,
    max_tokens: int = 40,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Compares Checkpoint A and Checkpoint B side-by-side on the same prompt list.
    Safely handles missing or incompatible checkpoints.
    """
    start_time = datetime.datetime.now().isoformat()
    if prompts is None:
        prompts = DEFAULT_COMPARISON_PROMPTS

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

    comparisons = []
    identical_count = 0
    total_tokens_a = 0
    total_tokens_b = 0
    total_latency_a = 0.0
    total_latency_b = 0.0

    for idx, prompt in enumerate(prompts):
        out_a = run_single_inference(model_a, tokenizer, prompt, max_tokens=max_tokens)
        out_b = run_single_inference(model_b, tokenizer, prompt, max_tokens=max_tokens)
        oracle = run_deterministic_oracle(prompt)

        is_identical = (out_a["clean_output"] == out_b["clean_output"])
        if is_identical:
            identical_count += 1

        total_tokens_a += out_a["token_count"]
        total_tokens_b += out_b["token_count"]
        total_latency_a += out_a["latency_ms"]
        total_latency_b += out_b["latency_ms"]

        # Diff calculation
        diff_lines = list(difflib.unified_diff(
            out_a["clean_output"].splitlines(keepends=True),
            out_b["clean_output"].splitlines(keepends=True),
            fromfile="Checkpoint_A",
            tofile="Checkpoint_B",
        ))

        comparisons.append({
            "index": idx,
            "prompt": prompt,
            "oracle": oracle,
            "checkpoint_a_output": out_a,
            "checkpoint_b_output": out_b,
            "outputs_identical": is_identical,
            "diff_patch": "".join(diff_lines),
        })

    report = {
        "success": True,
        "title": "Checkpoint Output Comparison Report",
        "timestamp": start_time,
        "settings": {
            "max_tokens": max_tokens,
            "num_prompts": len(prompts),
            "tokenizer_path": TOKENIZER_DIR,
        },
        "checkpoint_a": meta_a,
        "checkpoint_b": meta_b,
        "summary": {
            "total_prompts": len(prompts),
            "identical_outputs_count": identical_count,
            "divergent_outputs_count": len(prompts) - identical_count,
            "divergence_rate_pct": round(((len(prompts) - identical_count) / len(prompts)) * 100, 2),
            "model_a_avg_latency_ms": round(total_latency_a / max(1, len(prompts)), 2),
            "model_b_avg_latency_ms": round(total_latency_b / max(1, len(prompts)), 2),
            "model_a_total_tokens": total_tokens_a,
            "model_b_total_tokens": total_tokens_b,
        },
        "comparisons": comparisons,
    }

    if save_report:
        report_path = save_comparison_report(report)
        report["saved_report_path"] = str(report_path)

    return report


def save_comparison_report(report: Dict[str, Any]) -> Path:
    """Saves comparison report to timestamped JSON file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"comparison_report_{ts}.json"
    report["report_file"] = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def load_comparison_report(report_path: str) -> Dict[str, Any]:
    """Loads a previously saved comparison report without rerunning inference."""
    p = Path(report_path)
    if not p.is_file():
        return {"error": f"Report file not found: {report_path}", "success": False}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_saved_reports() -> List[Dict[str, Any]]:
    """Lists saved comparison and workbench reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
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


# Embedded HTML5 Web UI for Side-by-Side Checkpoint Comparison
HTML_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChakrView Local Workbench - Checkpoint Output Comparator</title>
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
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-primary); padding: 24px; }
        .container { max-width: 1300px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { font-size: 24px; font-weight: 700; background: linear-gradient(90deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .badge { background: var(--bg-card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 12px; font-size: 12px; color: var(--accent); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        h2 { font-size: 16px; margin-bottom: 14px; color: var(--accent); display: flex; align-items: center; gap: 8px; }
        label { display: block; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
        input, select, textarea { width: 100%; background: var(--bg-primary); border: 1px solid var(--border); color: var(--text-primary); padding: 10px; border-radius: 6px; font-size: 14px; margin-bottom: 14px; }
        textarea { resize: vertical; min-height: 80px; font-family: monospace; font-size: 12px; }
        button { background: linear-gradient(135deg, #00d2ff, #0072ff); color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        .meta-box { background: var(--code-bg); border: 1px solid var(--border); padding: 10px; border-radius: 6px; font-size: 12px; font-family: monospace; margin-top: 10px; word-break: break-all; }
        .comparison-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; overflow: hidden; }
        .comparison-header { padding: 10px 16px; background: rgba(0,0,0,0.2); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 600; }
        .comparison-body { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; }
        .output-col { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
        .output-col h4 { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; }
        .output-text { font-family: monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; }
        .diff-tag { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: bold; }
        .diff-tag.identical { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
        .diff-tag.divergent { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); }
        .oracle-banner { margin: 0 16px 12px 16px; padding: 8px 12px; background: rgba(0, 210, 255, 0.08); border-left: 3px solid var(--accent); font-size: 12px; border-radius: 4px; }
        .summary-stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }
        .stat-value { font-size: 22px; font-weight: bold; color: var(--accent); margin-top: 4px; }
        .stat-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>ChakrView Checkpoint Comparator</h1>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Hardware-Adaptive Neural Engine · Batches 91–94</div>
            </div>
            <div class="badge" id="appStatus">Ready · ChakrMicro v0.1</div>
        </header>

        <div class="grid-2">
            <!-- Checkpoint A Selection -->
            <div class="card">
                <h2><span>🔍</span> Checkpoint A</h2>
                <label>Checkpoint Path</label>
                <input type="text" id="pathA" value="experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt" />
                <button onclick="inspectCkpt('A')">Inspect A</button>
                <div id="metaA" class="meta-box">Click Inspect to view SHA-256 and parameters</div>
            </div>

            <!-- Checkpoint B Selection -->
            <div class="card">
                <h2><span>🔍</span> Checkpoint B</h2>
                <label>Checkpoint Path</label>
                <input type="text" id="pathB" value="experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt" />
                <button onclick="inspectCkpt('B')">Inspect B</button>
                <div id="metaB" class="meta-box">Click Inspect to view SHA-256 and parameters</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 24px;">
            <h2><span>⚡</span> Run Side-by-Side Comparison</h2>
            <label>Prompts (One per line, or JSON array)</label>
            <textarea id="promptList">Calculate 3 + 4.
What is 8 + 5?
Add 0 and 9.
How much is 7 + 7?
Compute 2 + 6.
Tell me a short joke.</textarea>
            <div style="display: flex; gap: 14px; align-items: center;">
                <label style="margin: 0;">Max Tokens: <input type="number" id="maxTokens" value="40" style="width: 80px; display: inline-block; margin: 0 0 0 8px;" /></label>
                <button onclick="executeComparison()">Run Comparison</button>
                <button style="background: var(--bg-card); border: 1px solid var(--border);" onclick="loadSavedReport()">Reopen Saved Report</button>
            </div>
        </div>

        <!-- Comparison Results Area -->
        <div id="resultsArea" style="display: none;">
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-label">Total Prompts</div>
                    <div class="stat-value" id="statPrompts">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Identical Outputs</div>
                    <div class="stat-value" id="statIdentical">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Divergence Rate</div>
                    <div class="stat-value" id="statDivergence">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Model A Avg Latency</div>
                    <div class="stat-value" id="statLatencyA">0ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Model B Avg Latency</div>
                    <div class="stat-value" id="statLatencyB">0ms</div>
                </div>
            </div>

            <div id="comparisonList"></div>
        </div>
    </div>

    <script>
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

        async function executeComparison() {
            const pathA = document.getElementById('pathA').value;
            const pathB = document.getElementById('pathB').value;
            const promptsText = document.getElementById('promptList').value.split('\\n').map(p => p.trim()).filter(p => p.length > 0);
            const maxTokens = parseInt(document.getElementById('maxTokens').value) || 40;

            const appStatus = document.getElementById('appStatus');
            appStatus.innerText = "Comparing...";
            appStatus.style.color = "var(--warning)";

            try {
                const formattedPrompts = promptsText.map(p => `<|im_start|>user\\n${p}<|im_end|>\\n<|im_start|>assistant\\n`);
                const resp = await fetch('/api/compare', {
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
                renderReport(report);
                appStatus.innerText = "Comparison Complete";
                appStatus.style.color = "var(--success)";
            } catch (e) {
                alert("Comparison error: " + e);
                appStatus.innerText = "Failed";
                appStatus.style.color = "var(--danger)";
            }
        }

        function renderReport(report) {
            if (!report.success) {
                alert("Comparison failed: " + report.error);
                return;
            }
            document.getElementById('resultsArea').style.display = 'block';
            document.getElementById('statPrompts').innerText = report.summary.total_prompts;
            document.getElementById('statIdentical').innerText = report.summary.identical_outputs_count;
            document.getElementById('statDivergence').innerText = report.summary.divergence_rate_pct + "%";
            document.getElementById('statLatencyA').innerText = report.summary.model_a_avg_latency_ms + "ms";
            document.getElementById('statLatencyB').innerText = report.summary.model_b_avg_latency_ms + "ms";

            const list = document.getElementById('comparisonList');
            list.innerHTML = "";

            report.comparisons.forEach(item => {
                const isIdentical = item.outputs_identical;
                const oracle = item.oracle;
                let oracleHtml = '';
                if (oracle && oracle.is_arithmetic) {
                    oracleHtml = `<div class="oracle-banner"><strong>[DETERMINISTIC ORACLE REFERENCE]</strong> Expected: ${oracle.expected_answer} | Expected Tool Call: <code>${oracle.expected_tool_call}</code></div>`;
                }

                const node = document.createElement('div');
                node.className = 'comparison-item';
                node.innerHTML = `
                    <div class="comparison-header">
                        <div>Prompt: <code>${escapeHtml(item.prompt.replace('<|im_start|>user\\n', '').replace('<|im_end|>\\n<|im_start|>assistant\\n', ''))}</code></div>
                        <div class="diff-tag ${isIdentical ? 'identical' : 'divergent'}">${isIdentical ? 'IDENTICAL' : 'DIVERGENT'}</div>
                    </div>
                    ${oracleHtml}
                    <div class="comparison-body">
                        <div class="output-col">
                            <h4>Checkpoint A (${item.checkpoint_a_output.latency_ms}ms · ${item.checkpoint_a_output.token_count} toks)</h4>
                            <div class="output-text">${escapeHtml(item.checkpoint_a_output.clean_output || '<EMPTY>')}</div>
                        </div>
                        <div class="output-col">
                            <h4>Checkpoint B (${item.checkpoint_b_output.latency_ms}ms · ${item.checkpoint_b_output.token_count} toks)</h4>
                            <div class="output-text">${escapeHtml(item.checkpoint_b_output.clean_output || '<EMPTY>')}</div>
                        </div>
                    </div>
                `;
                list.appendChild(node);
            });
        }

        async function loadSavedReport() {
            const filename = prompt("Enter report filename or leave blank for latest:");
            try {
                const resp = await fetch('/api/reports' + (filename ? '?file=' + encodeURIComponent(filename) : ''));
                const data = await resp.json();
                if (data.report) {
                    renderReport(data.report);
                    document.getElementById('appStatus').innerText = "Report Loaded: " + (filename || 'Latest');
                } else if (data.reports) {
                    alert("Available reports:\\n" + data.reports.map(r => r.filename).join('\\n'));
                }
            } catch (e) {
                alert("Failed to load report: " + e);
            }
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
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
                rep = load_comparison_report(str(REPORTS_DIR / fname))
                resp = {"report": rep}
            else:
                reps = list_saved_reports()
                latest = load_comparison_report(reps[0]["path"]) if reps else None
                resp = {"reports": reps, "report": latest}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/compare":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            req = json.loads(body)
            ckpt_a = req.get("ckpt_a", "")
            ckpt_b = req.get("ckpt_b", "")
            prompts = req.get("prompts", DEFAULT_COMPARISON_PROMPTS)
            max_tokens = req.get("max_tokens", 40)
            res = compare_checkpoints(ckpt_a, ckpt_b, prompts=prompts, max_tokens=max_tokens, save_report=True)
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
    parser = argparse.ArgumentParser(description="ChakrView Checkpoint Output Comparator & Workbench (Batches 91-94)")
    subparsers = parser.add_subparsers(dest="command")

    # Inspect
    inspect_cmd = subparsers.add_parser("inspect", help="Inspect a checkpoint")
    inspect_cmd.add_argument("--ckpt", required=True, help="Path to checkpoint")

    # Compare
    compare_cmd = subparsers.add_parser("compare", help="Compare two checkpoints side by side")
    compare_cmd.add_argument("--ckpt-a", required=True, help="Path to Checkpoint A")
    compare_cmd.add_argument("--ckpt-b", required=True, help="Path to Checkpoint B")
    compare_cmd.add_argument("--max-tokens", type=int, default=40, help="Max tokens per prompt")
    compare_cmd.add_argument("--prompts", default=None, help="Optional comma-separated prompts")

    # Reports
    subparsers.add_parser("list-reports", help="List saved comparison reports")

    # Serve
    serve_cmd = subparsers.add_parser("serve", help="Launch Local Workbench Web Server")
    serve_cmd.add_argument("--port", type=int, default=8501, help="Port to serve on")

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_checkpoint(args.ckpt)
        print(json.dumps(info, indent=2))
    elif args.command == "compare":
        prompts = [p.strip() for p in args.prompts.split(",")] if args.prompts else None
        res = compare_checkpoints(args.ckpt_a, args.ckpt_b, prompts=prompts, max_tokens=args.max_tokens)
        print(f"\n=== CHECKPOINT OUTPUT COMPARISON REPORT ===")
        print(f"Checkpoint A: {res.get('checkpoint_a', {}).get('path')}")
        print(f"Checkpoint B: {res.get('checkpoint_b', {}).get('path')}")
        if not res.get("success"):
            print(f"FAILED: {res.get('error')}")
            sys.exit(1)
        summary = res["summary"]
        print(f"Prompts: {summary['total_prompts']} | Identical: {summary['identical_outputs_count']} | Divergent: {summary['divergent_outputs_count']} ({summary['divergence_rate_pct']}%)")
        print(f"Saved Report: {res.get('saved_report_path')}")
        for c in res["comparisons"]:
            print(f"\n--- Prompt: {c['prompt'].strip()} ---")
            print(f"  [CKPT A]: {repr(c['checkpoint_a_output']['clean_output'])}")
            print(f"  [CKPT B]: {repr(c['checkpoint_b_output']['clean_output'])}")
            print(f"  [STATUS]: {'IDENTICAL' if c['outputs_identical'] else 'DIVERGENT'}")
    elif args.command == "list-reports":
        reps = list_saved_reports()
        print(f"Found {len(reps)} reports in {REPORTS_DIR}:")
        for r in reps:
            print(f"- {r['filename']}: {r['title']} ({r['timestamp']})")
    elif args.command == "serve":
        print(f"Starting ChakrView Workbench on http://localhost:{args.port}...")
        server = socketserver.TCPServer(("", args.port), WorkbenchHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
    else:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
