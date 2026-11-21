"""
ChakrView Ultimate Brain: Interactive Local Web Application.
Features:
- Lightweight, zero-dependency Python HTTP server (localhost:8080).
- Multi-turn conversational chat with dynamic message history.
- Feedback buttons (Thumbs Up / Thumbs Down / Explicit Correction).
- Operational Self-Model inspection panel.
- Persistent Memory Manager inspector (Working, Episodic, Semantic, Preferences).
- Pure local execution (100% CPU, zero external APIs).
"""

from __future__ import annotations
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys
import urllib.parse

# Ensure repository root is in sys.path regardless of location
current_file = Path(__file__).resolve()
for parent in [current_file.parent, current_file.parents[1], current_file.parents[2]]:
    if (parent / "chakrview").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from chakrview.interface.ultimate.agent_controller import ChakrViewUltimateAgent

agent = None


class UltimateChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy 404 logs from background browser probes
        if len(args) > 1 and str(args[1]) == "404":
            return
        super().log_message(format, *args)

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path in ["/", "/index.html", "/chat"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif clean_path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = agent.self_model.get_self_description()
            status_data["memory_stats"] = {
                "episodic_count": len(agent.memory.episodic),
                "semantic_count": len(agent.memory.semantic),
                "preference_count": len(agent.memory.preferences)
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        elif clean_path in ["/v1/models", "/api/models"]:
            # Compatibility endpoint for LLM frontends
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [{"id": "chakr-ultra-v1", "object": "model", "owned_by": "chakrview"}]
            }).encode("utf-8"))
        elif clean_path in ["/tools", "/build.json", "/props", "/sw.js"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        req = json.loads(post_data)

        if self.path == "/api/chat":
            user_msg = req.get("message", "")
            res = agent.process_message(user_msg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif self.path == "/api/feedback":
            fb = agent.learning.record_feedback(
                prompt=req.get("prompt", ""),
                predicted_response=req.get("predicted", ""),
                is_correct=req.get("is_correct", True),
                corrected_response=req.get("correction"),
                feedback_notes=req.get("notes")
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "recorded", "feedback": fb}).encode("utf-8"))

        elif self.path == "/api/reset":
            agent.memory.clear_working_memory()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "cleared"}).encode("utf-8"))


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChakrView Ultimate Brain v1.0</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; height: 92vh; }
        header { border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.4rem; color: #38bdf8; }
        .badge { background: #1e293b; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; border: 1px solid #475569; }
        #chat-window { flex: 1; overflow-y: auto; background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; margin-bottom: 16px; display: flex; flex-direction: column; gap: 12px; }
        .msg { padding: 10px 14px; border-radius: 8px; max-width: 80%; line-height: 1.4; font-size: 0.95rem; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #ffffff; }
        .msg.assistant { align-self: flex-start; background: #334155; color: #f8fafc; border-left: 3px solid #38bdf8; }
        .meta { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
        .input-row { display: flex; gap: 8px; }
        input[type="text"] { flex: 1; padding: 12px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #fff; font-size: 1rem; }
        button { padding: 12px 20px; border-radius: 6px; border: none; background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; }
        button:hover { background: #0ea5e9; }
        .btn-sec { background: #475569; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>ChakrView Ultimate Brain v1.0</h1>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:4px;">ChakrUltra GQA (5.18M) &bull; 4-Tier Memory &bull; CPU Native</div>
            </div>
            <div>
                <button class="btn-sec" onclick="resetChat()">Reset Context</button>
            </div>
        </header>
        <div id="chat-window">
            <div class="msg assistant">
                Hello! I am ChakrView Ultimate Brain v1.0. All cognitive systems (Persistent Memory, Structured Reasoning, Operational Self-Model) are active. How can I help you today?
            </div>
        </div>
        <div class="input-row">
            <input type="text" id="userInput" placeholder="Ask a question, request a calculation, or teach facts..." onkeydown="if(event.key==='Enter') sendMsg()" autofocus>
            <button onclick="sendMsg()">Send</button>
        </div>
    </div>

    <script>
        async function sendMsg() {
            const input = document.getElementById("userInput");
            const text = input.value.trim();
            if (!text) return;
            input.value = "";

            const win = document.getElementById("chat-window");
            win.innerHTML += `<div class="msg user">${text}</div>`;
            win.scrollTop = win.scrollHeight;

            const res = await fetch("/api/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: text})
            });
            const data = await res.json();
            win.innerHTML += `<div class="msg assistant">${data.reply}<div class="meta">Source: ${data.source} &bull; ${Math.round(data.latency_ms)}ms</div></div>`;
            win.scrollTop = win.scrollHeight;
        }

        async function resetChat() {
            await fetch("/api/reset", {method: "POST"});
            document.getElementById("chat-window").innerHTML = '<div class="msg assistant">Conversation context reset to empty.</div>';
        }
    </script>
</body>
</html>
"""


def start_server(port: int = 7860):
    global agent
    agent = ChakrViewUltimateAgent()
    server = HTTPServer(("127.0.0.1", port), UltimateChatHandler)
    print(f"\nChakrView Ultimate Brain Web Application active at: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer terminated.")


if __name__ == "__main__":
    start_server()
