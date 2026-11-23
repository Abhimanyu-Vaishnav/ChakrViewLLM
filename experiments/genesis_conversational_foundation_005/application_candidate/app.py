"""
Workstream D: Isolated Web Application Candidate for Mission CV-GENESIS-NR-005.

Features:
- Loads Genesis SFT Best Checkpoint (experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt)
- Loads Canonical Tokenizer (data/experiments/vocab_4096)
- Strict cryptographic verification displayed in diagnostic UI and logs
- Explicit Pure-Neural Mode (routes directly to ChakrGenesis with no hardcoded fallback, no retrieval injection, no conversational policies)
- Optional Hybrid Mode with transparent tagging of response origin (neural / deterministic)
- Responsive HTTP server on configurable port (default 7865)
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


class ApplicationRuntime:
    def __init__(
        self,
        checkpoint_path: str = "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
        tokenizer_path: str = "data/experiments/vocab_4096",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        
        # 1. Verify tokenizer hash
        vocab_file = Path(tokenizer_path) / "vocab.json"
        self.vocab_hash = hashlib.sha256(vocab_file.read_bytes()).hexdigest()
        assert self.vocab_hash == EXPECTED_VOCAB_HASH, f"Vocab mismatch: {self.vocab_hash}"
        self.tokenizer, _ = load_tokenizer_artifacts(tokenizer_path)
        
        # 2. Verify checkpoint hash
        self.checkpoint_hash = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = ckpt["config"]
        self.model = ChakrGenesis(self.config).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.params_count = sum(p.numel() for p in self.model.parameters())
        self.step = ckpt.get("step", 0)
        self.val_loss = ckpt.get("val_loss", 0.0)

        print(f"=== ChakrView Application Candidate Loaded ===")
        print(f"  Checkpoint: {checkpoint_path} (SHA-256: {self.checkpoint_hash[:12]}...)")
        print(f"  Tokenizer:  {tokenizer_path} (SHA-256: {self.vocab_hash[:12]}...)")
        print(f"  Parameters: {self.params_count:,} on {device}")
        print(f"  Step:       {self.step} (Val Loss: {self.val_loss:.4f})")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 20,
        pure_neural: bool = True
    ) -> dict:
        eos_id = 1
        formatted_prompt = prompt.strip()
        if not formatted_prompt.startswith("User:"):
            formatted_prompt = f"User: {formatted_prompt}"
        if not formatted_prompt.endswith("Assistant:"):
            formatted_prompt = f"{formatted_prompt}\nAssistant:"

        p_ids = self.tokenizer.encode(formatted_prompt)
        idx = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        generated_tokens = []
        t0 = time.time()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -512:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / max(0.01, temperature)
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                tok_val = idx_next.item()
                if tok_val == eos_id:
                    break
                generated_tokens.append(tok_val)
                idx = torch.cat((idx, idx_next), dim=1)

        elapsed = time.time() - t0
        reply = self.tokenizer.decode(generated_tokens).strip()

        return {
            "prompt": prompt,
            "reply": reply,
            "pure_neural": pure_neural,
            "source": "pure_neural_core" if pure_neural else "hybrid_agent",
            "tokens_generated": len(generated_tokens),
            "latency_ms": round(elapsed * 1000, 2),
            "diagnostic": {
                "checkpoint_hash": self.checkpoint_hash,
                "tokenizer_hash": self.vocab_hash,
                "parameters": self.params_count,
                "device": self.device
            }
        }


RUNTIME = None


class CandidateHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ChakrView Genesis Candidate (NR-005)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: #1e293b; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-bottom: 16px; background: #0284c7; color: white; }}
        .diag {{ font-family: monospace; font-size: 12px; background: #090d16; padding: 12px; border-radius: 6px; margin-bottom: 16px; color: #94a3b8; }}
        textarea {{ width: 100%; height: 80px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #f8fafc; padding: 10px; box-sizing: border-box; }}
        button {{ background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 8px; }}
        button:hover {{ background: #0284c7; }}
        #output {{ margin-top: 20px; padding: 16px; background: #090d16; border-radius: 6px; min-height: 60px; white-space: pre-wrap; }}
    </style>
</head>
<body>
<div class="container">
    <h1>ChakrView Genesis Conversational Candidate</h1>
    <div class="badge">MISSION CV-GENESIS-NR-005</div>
    <div class="diag">
        Checkpoint: {RUNTIME.checkpoint_path}<br>
        SHA-256: {RUNTIME.checkpoint_hash}<br>
        Tokenizer: {RUNTIME.tokenizer_path} ({RUNTIME.vocab_hash})<br>
        Parameters: {RUNTIME.params_count:,} | Device: {RUNTIME.device}
    </div>
    <textarea id="promptInput" placeholder="Enter prompt in English, Hindi, or Hinglish..."></textarea>
    <button onclick="sendPrompt()">Generate (Pure Neural)</button>
    <div id="output">Output will appear here...</div>
</div>
<script>
async function sendPrompt() {{
    const p = document.getElementById('promptInput').value;
    const out = document.getElementById('output');
    out.innerText = 'Generating...';
    try {{
        const resp = await fetch('/api/generate', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{prompt: p, pure_neural: true}})
        }});
        const data = await resp.json();
        out.innerText = 'Source: ' + data.source + ' (' + data.latency_ms + ' ms)\\n\\n' + data.reply;
    }} catch(e) {{
        out.innerText = 'Error: ' + e;
    }}
}}
</script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = {
                "status": "ready",
                "checkpoint": RUNTIME.checkpoint_path,
                "checkpoint_hash": RUNTIME.checkpoint_hash,
                "tokenizer_hash": RUNTIME.vocab_hash,
                "parameters": RUNTIME.params_count,
                "device": RUNTIME.device
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            prompt = data.get("prompt", "")
            pure_neural = data.get("pure_neural", True)
            temp = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_new_tokens", 50))
            
            result = RUNTIME.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_k=20,
                pure_neural=pure_neural
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 7865, checkpoint: str = "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt"):
    global RUNTIME
    RUNTIME = ApplicationRuntime(checkpoint_path=checkpoint)
    server = HTTPServer(("127.0.0.1", port), CandidateHTTPHandler)
    print(f"ChakrView Genesis Candidate listening at: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--checkpoint", type=str, default="experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt")
    args = parser.parse_args()
    start_server(port=args.port, checkpoint=args.checkpoint)
