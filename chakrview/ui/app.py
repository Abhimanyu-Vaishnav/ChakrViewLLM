"""
ChakrView Local Web Application Server.
Mission CV-GENESIS-NR-012.

Zero external dependencies: Built entirely on Python's standard library (http.server).
Features:
- High-velocity local HTTP server for ChakrView 17M.
- Responsive, dark-theme modern AI console UI.
- Interactive mode toggle: Pure Neural Core vs Hybrid Cognitive System.
- Working memory inspection panel (visualizing asserted entity facts).
- Memory clear endpoint and session state.
- Real-time latency and token generation metrics.
- Endpoints:
    GET  /
    GET  /api/health
    GET  /api/memory
    POST /api/mode
    POST /api/memory/clear
    POST /api/chat
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.runtime.conversation_session import ConversationSession

CHECKPOINT_PATH = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
VOCAB_DIR = "data/experiments/vocab_4096"

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ChakrView 17M Local AI Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --accent: #58a6ff;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --green: #3fb950;
      --orange: #d29922;
      --purple: #bc8cff;
      --user-bubble: #1f6feb22;
      --assistant-bubble: #21262d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
    
    #sidebar {
      width: 320px; background: var(--panel); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; padding: 1.2rem;
    }
    .brand { font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 0.2rem; display: flex; align-items: center; gap: 8px; }
    .badge { font-size: 0.75rem; background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; }
    .subtitle { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1.5rem; }
    
    .section-title { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.6rem; letter-spacing: 0.5px; }
    
    .mode-selector { display: flex; gap: 8px; margin-bottom: 1.5rem; }
    .mode-btn {
      flex: 1; padding: 8px; background: #21262d; border: 1px solid var(--border);
      color: var(--text); border-radius: 6px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s;
    }
    .mode-btn.active { background: #1f6feb; border-color: #58a6ff; color: #fff; font-weight: 600; }
    
    .memory-box {
      flex: 1; background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
      padding: 0.8rem; overflow-y: auto; font-size: 0.85rem; margin-bottom: 1rem;
    }
    .fact-item { margin-bottom: 6px; padding: 4px 6px; background: #161b22; border-left: 3px solid var(--purple); border-radius: 3px; font-size: 0.8rem; }
    .empty-facts { color: var(--text-muted); font-style: italic; font-size: 0.8rem; }
    
    .sidebar-actions { display: flex; flex-direction: column; gap: 8px; }
    .btn-secondary {
      padding: 8px; background: #21262d; border: 1px solid var(--border);
      color: var(--text); border-radius: 6px; cursor: pointer; font-size: 0.85rem;
    }
    .btn-secondary:hover { background: #30363d; }
    
    #chat-container { flex: 1; display: flex; flex-direction: column; height: 100vh; }
    #header {
      height: 55px; border-bottom: 1px solid var(--border); display: flex; align-items: center;
      justify-content: space-between; padding: 0 1.5rem; background: var(--panel);
    }
    .status-indicator { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: var(--text-muted); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    
    #messages { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .msg { display: flex; flex-direction: column; max-width: 80%; line-height: 1.5; }
    .msg.user { align-self: flex-end; }
    .msg.assistant { align-self: flex-start; }
    
    .bubble {
      padding: 10px 14px; border-radius: 8px; font-size: 0.92rem; word-break: break-word;
    }
    .msg.user .bubble { background: var(--user-bubble); border: 1px solid #1f6feb55; color: #fff; }
    .msg.assistant .bubble { background: var(--assistant-bubble); border: 1px solid var(--border); }
    
    .meta-tag {
      font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; display: flex; gap: 10px; align-items: center;
    }
    .tag-origin {
      padding: 1px 6px; border-radius: 4px; font-weight: 600; text-transform: uppercase; font-size: 0.68rem;
    }
    .tag-origin.pure_neural { background: #1f6feb33; color: var(--accent); }
    .tag-origin.episodic_memory { background: #bc8cff33; color: var(--purple); }
    .tag-origin.deterministic_math { background: #d2992233; color: var(--orange); }
    .tag-origin.principled_abstention { background: #da363333; color: #f85149; }
    
    #input-bar {
      padding: 1.2rem; border-top: 1px solid var(--border); background: var(--panel);
      display: flex; gap: 10px; align-items: center;
    }
    #prompt-input {
      flex: 1; padding: 10px 14px; background: #0d1117; border: 1px solid var(--border);
      border-radius: 6px; color: #fff; font-size: 0.95rem; outline: none;
    }
    #prompt-input:focus { border-color: var(--accent); }
    #send-btn {
      padding: 10px 20px; background: #238636; border: none; border-radius: 6px;
      color: #fff; font-weight: 600; cursor: pointer; transition: background 0.2s;
    }
    #send-btn:hover { background: #2ea043; }
    #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="brand">
      ChakrView <span class="badge">17M Stage-D</span>
    </div>
    <div class="subtitle">Mission CV-GENESIS-NR-012</div>
    
    <div class="section-title">Inference Execution Mode</div>
    <div class="mode-selector">
      <button id="btn-hybrid" class="mode-btn active" onclick="setMode('hybrid')">Hybrid Cognitive</button>
      <button id="btn-neural" class="mode-btn" onclick="setMode('pure_neural')">Pure Neural Core</button>
    </div>
    
    <div class="section-title">Cognitive Working Memory</div>
    <div id="memory-list" class="memory-box">
      <div class="empty-facts">No facts stored yet.</div>
    </div>
    
    <div class="sidebar-actions">
      <button class="btn-secondary" onclick="clearMemory()">Reset Working Memory</button>
      <button class="btn-secondary" onclick="refreshMemory()">Refresh Memory View</button>
    </div>
  </div>

  <div id="chat-container">
    <div id="header">
      <div class="status-indicator">
        <span class="status-dot"></span>
        <span id="backend-status">Connected to Local Neural Engine</span>
      </div>
      <div style="font-size:0.8rem; color:var(--text-muted)">
        Parameters: 17,063,808 | Vocab: 4,096 BPE
      </div>
    </div>

    <div id="messages">
      <div class="msg assistant">
        <div class="bubble">
          Welcome to ChakrView 17M. You can query in English, Hindi, or Hinglish. Toggle between <b>Hybrid Cognitive</b> mode (with working memory and deterministic tools) and <b>Pure Neural Core</b> mode on the left.
        </div>
      </div>
    </div>

    <form id="input-bar" onsubmit="sendMessage(event)">
      <input type="text" id="prompt-input" placeholder="Type your message here..." autocomplete="off" />
      <button type="submit" id="send-btn">Send</button>
    </form>
  </div>

  <script>
    let activeMode = 'hybrid';

    async function setMode(mode) {
      activeMode = mode;
      document.getElementById('btn-hybrid').classList.toggle('active', mode === 'hybrid');
      document.getElementById('btn-neural').classList.toggle('active', mode === 'pure_neural');
      await fetch('/api/mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: mode})
      });
    }

    async function refreshMemory() {
      const res = await fetch('/api/memory');
      const data = await res.json();
      const box = document.getElementById('memory-list');
      box.innerHTML = '';
      const facts = Object.entries(data.stored_facts || {});
      if (facts.length === 0) {
        box.innerHTML = '<div class="empty-facts">No facts stored yet.</div>';
      } else {
        facts.forEach(([k, v]) => {
          const div = document.createElement('div');
          div.className = 'fact-item';
          div.innerHTML = `<strong>${k}:</strong> ${v}`;
          box.appendChild(div);
        });
      }
    }

    async function clearMemory() {
      await fetch('/api/memory/clear', {method: 'POST'});
      refreshMemory();
    }

    async function sendMessage(e) {
      e.preventDefault();
      const input = document.getElementById('prompt-input');
      const text = input.value.trim();
      if (!text) return;
      
      input.value = '';
      const sendBtn = document.getElementById('send-btn');
      sendBtn.disabled = true;

      // Add user message
      appendMessage('user', text);

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: text, mode: activeMode})
        });
        const data = await res.json();
        appendMessage('assistant', data.response, data);
        refreshMemory();
      } catch (err) {
        appendMessage('assistant', 'Error communicating with engine: ' + err.message);
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    function appendMessage(role, text, meta) {
      const container = document.getElementById('messages');
      const msgDiv = document.createElement('div');
      msgDiv.className = `msg ${role}`;
      
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerText = text;
      msgDiv.appendChild(bubble);

      if (meta && role === 'assistant') {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta-tag';
        const originClass = (meta.origin || 'neural').replace(/[^a-zA-Z0-9_]/g, '');
        let speedStr = meta.tokens_per_sec ? ` | ${meta.tokens_per_sec.toFixed(1)} tok/s` : '';
        let latStr = meta.latency_sec ? `${(meta.latency_sec * 1000).toFixed(0)} ms` : '';
        metaDiv.innerHTML = `<span class="tag-origin ${originClass}">${meta.origin || 'neural'}</span> <span>${latStr}${speedStr}</span>`;
        msgDiv.appendChild(metaDiv);
      }

      container.appendChild(msgDiv);
      container.scrollTop = container.scrollHeight;
    }

    // Initial memory fetch
    refreshMemory();
  </script>
</body>
</html>
"""


class ChakrViewServerState:
    _engine: Optional[ChakrViewInferenceEngine17M] = None
    _session: Optional[ConversationSession] = None

    @classmethod
    def get_session(cls) -> ConversationSession:
        if cls._engine is None:
            cls._engine = ChakrViewInferenceEngine17M(CHECKPOINT_PATH, VOCAB_DIR)
        if cls._session is None:
            cls._session = ConversationSession(cls._engine, mode="hybrid")
        return cls._session


class ChakrViewHTTPHandler(BaseHTTPRequestHandler):

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            sess = ChakrViewServerState.get_session()
            self._send_json({
                "status": "healthy",
                "model": "ChakrGenesis-17M",
                "device": sess.engine.device.type,
                "checkpoint": str(sess.engine.checkpoint_path),
                "mode": sess.mode
            })
        elif self.path == "/api/memory":
            sess = ChakrViewServerState.get_session()
            self._send_json(sess.get_memory_state())
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/mode":
            data = self._read_json()
            sess = ChakrViewServerState.get_session()
            mode = data.get("mode", "hybrid")
            sess.set_mode(mode)
            self._send_json({"status": "success", "mode": sess.mode})
        elif self.path == "/api/memory/clear":
            sess = ChakrViewServerState.get_session()
            sess.clear_memory()
            self._send_json({"status": "success", "message": "Memory cleared."})
        elif self.path == "/api/chat":
            data = self._read_json()
            sess = ChakrViewServerState.get_session()
            req_mode = data.get("mode")
            if req_mode and req_mode in ["pure_neural", "hybrid"]:
                sess.set_mode(req_mode)
            
            msg = data.get("message", "")
            temp = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.9))
            top_k = int(data.get("top_k", 40))
            max_tokens = int(data.get("max_tokens", 128))

            result = sess.chat(
                user_message=msg,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_tokens
            )
            self._send_json(result)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Mute standard access log output to keep terminal clean
        pass


def run_server(host: str = "127.0.0.1", port: int = 8000):
    server = HTTPServer((host, port), ChakrViewHTTPHandler)
    print(f"ChakrView Web Console running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
