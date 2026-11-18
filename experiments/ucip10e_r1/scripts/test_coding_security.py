import sys
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

def test_payload(name: str, payload_code: str):
    full_code = payload_code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        res = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        passed = (res.returncode == 0)
        return {
            "name": name,
            "returncode": res.returncode,
            "stdout": res.stdout[:200],
            "stderr": res.stderr[:200],
            "executed": passed
        }
    except subprocess.TimeoutExpired:
        return {"name": name, "returncode": -1, "stdout": "", "stderr": "TIMEOUT", "executed": False}
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

# Security Probes (Harmless read / env / process checks)
probes = {
    "filesystem_read_outside_workspace": "import os; print('ENV_USER:', os.environ.get('USERNAME'))",
    "network_socket_creation": "import socket; s = socket.socket(); print('Socket created')",
    "infinite_loop_timeout": "while True: pass",
    "file_write_in_temp": "with open('probe_test.tmp', 'w') as f: f.write('test'); import os; os.remove('probe_test.tmp'); print('File write successful')"
}

results = {}
for name, code in probes.items():
    results[name] = test_payload(name, code)
    print(f"Probe '{name}': returncode={results[name]['returncode']}, stderr={repr(results[name]['stderr'])}")

import json
with open(REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "coding_security_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
