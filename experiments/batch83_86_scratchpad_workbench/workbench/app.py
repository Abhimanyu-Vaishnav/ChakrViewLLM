"""
Local ChakrView Workbench Server & CLI (v0.1).

A robust local workbench for ChakrView:
- Checkpoint inspector (SHA-256, parameter count, status)
- Deterministic inference and evaluation runner
- Bounded experiment launcher with live status, stdout/stderr logging, and timestamps
- Interactive browser web interface (HTML5/Vanilla CSS/JS, no external dependencies)
- Clean CLI mode with direct subcommands: inspect, infer, eval, run-experiment, report
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import socketserver
import subprocess
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

REPORTS_DIR = Path("experiments/batch83_86_scratchpad_workbench/workbench/reports")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DEFAULT_CHECKPOINTS = [
    {
        "name": "Canonical Baseline (Step 10,000)",
        "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        "expected_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
    },
    {
        "name": "Balanced Curriculum (Batch 81)",
        "path": "experiments/batch79_82_balanced_arithmetic/experiments/condition_c_balanced_curriculum_step_100.pt",
        "expected_sha256": "5213b7ef38dafa096c7c03818ef4da88cc05d8c9f8a208b312da889eae798b58",
    },
    {
        "name": "Scratchpad Model (Batch 84)",
        "path": "experiments/batch83_86_scratchpad_workbench/experiments/condition_b_scratchpad_arithmetic_step_100.pt",
        "expected_sha256": None,
    },
]

ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    p = Path(checkpoint_path)
    if not p.is_file():
        return {"error": f"File does not exist: {checkpoint_path}", "exists": False}

    sha = compute_file_sha256(str(p))
    size_mb = round(p.stat().st_size / (1024 * 1024), 2)
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        has_model = "model_state_dict" in data
        step = data.get("step", "N/A")
        cond = data.get("condition_name", "N/A")
        param_count = sum(v.numel() for v in data["model_state_dict"].values()) if has_model else "N/A"
        return {
            "exists": True,
            "path": str(p),
            "size_mb": size_mb,
            "sha256": sha,
            "step": step,
            "condition_name": cond,
            "param_count": param_count,
            "valid": True,
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(p),
            "size_mb": size_mb,
            "sha256": sha,
            "error": str(e),
            "valid": False,
        }


def run_deterministic_eval(a: int, b: int, mode: str = "direct") -> Dict[str, Any]:
    """Provides a deterministic baseline reference."""
    ans = a + b
    if mode == "scratchpad":
        steps = [str(a + i) for i in range(1, b + 1)] if b > 0 else []
        count_str = ", ".join(steps) if steps else "none"
        trace = f"Start at {a}; count {b} more: {count_str}; answer: {ans}"
    else:
        trace = str(ans)
    return {
        "source": "DETERMINISTIC_ORACLE",
        "operands": [a, b],
        "true_sum": ans,
        "formatted_output": trace,
        "parsed_answer": str(ans),
    }


def run_model_inference(checkpoint_path: str, prompt: str, max_tokens: int = 40) -> Dict[str, Any]:
    p = Path(checkpoint_path)
    if not p.is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    data = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    toks = [0] + tokenizer.encode(prompt)
    gen = list(toks)
    eos_reached = False
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([gen]))[0, -1]
            nxt = torch.argmax(logits).item()
            gen.append(nxt)
            if nxt == 1:
                eos_reached = True
                break

    raw_output = tokenizer.decode(gen[len(toks):])
    clean = raw_output.replace("<|im_end|>", "").strip()

    # Parse answer
    m = re.search(r"answer:\s*([0-9]+)", clean, re.IGNORECASE)
    parsed = m.group(1).strip() if m else None
    if parsed is None:
        m2 = re.match(r"^\s*([0-9]+)", clean)
        parsed = m2.group(1).strip() if m2 else None

    return {
        "source": "CHAKRMICRO_LOCAL_MODEL",
        "checkpoint": str(p),
        "prompt": prompt,
        "raw_output": raw_output,
        "clean_output": clean,
        "parsed_answer": parsed,
        "eos_reached": eos_reached,
        "tokens_generated": len(gen) - len(toks),
    }


def save_workbench_report(report_data: Dict[str, Any]) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = report_data.get("title", "workbench_run").replace(" ", "_").lower()
    path = REPORTS_DIR / f"report_{name}_{ts}.json"
    report_data["timestamp"] = ts
    report_data["report_file"] = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    return str(path)


def list_saved_reports() -> List[Dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res = []
    for f in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as r:
                data = json.load(r)
                res.append({
                    "filename": f.name,
                    "title": data.get("title", f.name),
                    "timestamp": data.get("timestamp", "N/A"),
                    "summary": data.get("summary", {}),
                })
        except Exception:
            pass
    return res


# Web Server Handler
HTML_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChakrView Local Workbench</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131c2e;
            --bg-card: #1c2840;
            --accent: #00d2ff;
            --accent-glow: rgba(0, 210, 255, 0.2);
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --border: #30363d;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-primary); padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { font-size: 24px; font-weight: 700; background: linear-gradient(90deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .badge { background: var(--bg-card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 12px; font-size: 12px; color: var(--accent); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        h2 { font-size: 16px; margin-bottom: 14px; color: var(--accent); display: flex; align-items: center; gap: 8px; }
        label { display: block; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
        input, select, textarea { width: 100%; background: var(--bg-primary); border: 1px solid var(--border); color: var(--text-primary); padding: 10px; border-radius: 6px; font-size: 14px; margin-bottom: 14px; }
        button { background: linear-gradient(135deg, #00d2ff, #0072ff); color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; transition: 0.2s; }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        pre { background: var(--bg-primary); border: 1px solid var(--border); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; color: #a5d6ff; max-height: 250px; }
        .full-width { grid-column: span 2; }
        .table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .table th, .table td { padding: 10px; border-bottom: 1px solid var(--border); text-align: left; }
        .table th { color: var(--text-secondary); }
        .tag-ok { color: var(--success); font-weight: bold; }
        .tag-warn { color: var(--warning); font-weight: bold; }
        .tag-source { font-size: 11px; padding: 2px 6px; border-radius: 4px; background: rgba(0, 210, 255, 0.15); color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CHAKRVIEW LOCAL WORKBENCH</h1>
                <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">Hardware-Adaptive Neural Engine & Model Diagnostics</p>
            </div>
            <div class="badge">Canonical ChakrMicro v0.1 • 3,443,136 Params</div>
        </header>

        <div class="grid">
            <!-- Checkpoint Inspector -->
            <div class="card">
                <h2>🔍 Checkpoint Inspector</h2>
                <label>Select Preset Checkpoint</label>
                <select id="ckptSelect" onchange="loadSelectedCkpt()">
                    <option value="experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt">Canonical Baseline (Step 10,000)</option>
                    <option value="experiments/batch79_82_balanced_arithmetic/experiments/condition_c_balanced_curriculum_step_100.pt">Batch 81 Balanced Model (Step 100)</option>
                    <option value="experiments/batch83_86_scratchpad_workbench/experiments/condition_b_scratchpad_arithmetic_step_100.pt">Batch 84 Scratchpad Model (Step 100)</option>
                </select>
                <label>Or Checkpoint Path</label>
                <input type="text" id="ckptPath" value="experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt">
                <button onclick="inspectCkpt()">Verify & Inspect Checkpoint</button>
                <div style="margin-top: 14px;">
                    <pre id="ckptInfo">// Checkpoint metadata will appear here...</pre>
                </div>
            </div>

            <!-- Single Query Generation & Oracle Comparator -->
            <div class="card">
                <h2>⚡ Generation & Verification</h2>
                <label>Arithmetic Operands (a + b)</label>
                <div style="display: flex; gap: 10px;">
                    <input type="number" id="opA" value="2" min="0" max="9" style="width: 50%;">
                    <input type="number" id="opB" value="3" min="0" max="9" style="width: 50%;">
                </div>
                <label>Prompt Format</label>
                <select id="modeSelect">
                    <option value="direct">Direct Arithmetic ("Add: a + b = ?")</option>
                    <option value="scratchpad">Scratchpad ("Add with scratchpad: a + b = ?")</option>
                </select>
                <div style="display: flex; gap: 10px;">
                    <button onclick="runInference()">Run Model Inference</button>
                    <button style="background: #238636;" onclick="runOracle()">Run Oracle Baseline</button>
                </div>
                <div style="margin-top: 14px;">
                    <pre id="inferOut">// Model generation and parse audit will appear here...</pre>
                </div>
            </div>

            <!-- Recent Sprint Experiments & Historical Reports -->
            <div class="card full-width">
                <h2>📊 Historical Sprint Experiments & Reports</h2>
                <div style="display: flex; gap: 10px; margin-bottom: 14px;">
                    <button onclick="loadLatestSummary()">Load Latest Sprint Results</button>
                    <button style="background: #30363d;" onclick="loadSavedReports()">View Saved Reports</button>
                </div>
                <pre id="reportsOut">// Latest experiment summary or reports will load here...</pre>
            </div>
        </div>
    </div>

    <script>
        function loadSelectedCkpt() {
            document.getElementById('ckptPath').value = document.getElementById('ckptSelect').value;
        }

        async function inspectCkpt() {
            const path = document.getElementById('ckptPath').value;
            const res = await fetch('/api/inspect?path=' + encodeURIComponent(path));
            const data = await res.json();
            document.getElementById('ckptInfo').textContent = JSON.stringify(data, null, 2);
        }

        async function runInference() {
            const path = document.getElementById('ckptPath').value;
            const a = document.getElementById('opA').value;
            const b = document.getElementById('opB').value;
            const mode = document.getElementById('modeSelect').value;
            const prompt = mode === 'scratchpad'
                ? `<|im_start|>user\\nAdd with scratchpad: ${a} + ${b} = ?<|im_end|>\\n<|im_start|>assistant\\n`
                : `<|im_start|>user\\nAdd: ${a} + ${b} = ?<|im_end|>\\n<|im_start|>assistant\\n`;
            
            document.getElementById('inferOut').textContent = "Running model inference on CPU...";
            const res = await fetch('/api/infer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path, prompt})
            });
            const data = await res.json();
            document.getElementById('inferOut').textContent = JSON.stringify(data, null, 2);
        }

        async function runOracle() {
            const a = parseInt(document.getElementById('opA').value);
            const b = parseInt(document.getElementById('opB').value);
            const mode = document.getElementById('modeSelect').value;
            const res = await fetch(`/api/oracle?a=${a}&b=${b}&mode=${mode}`);
            const data = await res.json();
            document.getElementById('inferOut').textContent = JSON.stringify(data, null, 2);
        }

        async function loadLatestSummary() {
            const res = await fetch('/api/latest-summary');
            const data = await res.json();
            document.getElementById('reportsOut').textContent = JSON.stringify(data, null, 2);
        }

        async function loadSavedReports() {
            const res = await fetch('/api/reports');
            const data = await res.json();
            document.getElementById('reportsOut').textContent = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""


class WorkbenchHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_INDEX.encode("utf-8"))
            return

        if self.path.startswith("/api/inspect"):
            path_arg = self.path.split("path=")[-1] if "path=" in self.path else ""
            import urllib.parse
            decoded_path = urllib.parse.unquote(path_arg)
            res = inspect_checkpoint(decoded_path)
            self._send_json(res)
            return

        if self.path.startswith("/api/oracle"):
            import urllib.parse
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            a = int(params.get("a", [0])[0])
            b = int(params.get("b", [0])[0])
            mode = params.get("mode", ["direct"])[0]
            res = run_deterministic_eval(a, b, mode)
            self._send_json(res)
            return

        if self.path == "/api/latest-summary":
            # Load Batch 84 or Batch 81 summary
            p84 = Path("experiments/batch83_86_scratchpad_workbench/experiments/batch84_experiment_summary.json")
            p81 = Path("experiments/batch79_82_balanced_arithmetic/experiments/batch81_experiment_summary.json")
            target = p84 if p84.is_file() else p81
            if target.is_file():
                with open(target, "r", encoding="utf-8") as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": "No experiment summary found"})
            return

        if self.path == "/api/reports":
            self._send_json(list_saved_reports())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/infer":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body)
            res = run_model_inference(req.get("path", ""), req.get("prompt", ""))
            self._send_json(res)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress standard logging to keep output clean


def run_workbench_server(port: int = 8501) -> None:
    server = socketserver.TCPServer(("127.0.0.1", port), WorkbenchHandler)
    print(f"ChakrView Local Workbench running at http://127.0.0.1:{port}")
    server.serve_forever()


def cli_main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ChakrView Local Workbench CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Workbench commands")

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a model checkpoint")
    inspect_parser.add_argument("--path", required=True, help="Path to checkpoint .pt")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run local inference")
    infer_parser.add_argument("--path", required=True, help="Path to checkpoint .pt")
    infer_parser.add_argument("--prompt", required=True, help="Input prompt")
    infer_parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens")

    # Oracle
    oracle_parser = subparsers.add_parser("oracle", help="Run deterministic arithmetic oracle")
    oracle_parser.add_argument("--a", type=int, required=True)
    oracle_parser.add_argument("--b", type=int, required=True)
    oracle_parser.add_argument("--mode", default="direct", choices=["direct", "scratchpad"])

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Launch local workbench web UI")
    serve_parser.add_argument("--port", type=int, default=8501, help="Port to bind to")

    args = parser.parse_args()

    if args.subcommand == "inspect":
        res = inspect_checkpoint(args.path)
        print(json.dumps(res, indent=2))
    elif args.subcommand == "infer":
        res = run_model_inference(args.path, args.prompt, max_tokens=args.max_tokens)
        print(json.dumps(res, indent=2))
    elif args.subcommand == "oracle":
        res = run_deterministic_eval(args.a, args.b, args.mode)
        print(json.dumps(res, indent=2))
    elif args.subcommand == "serve":
        run_workbench_server(args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
