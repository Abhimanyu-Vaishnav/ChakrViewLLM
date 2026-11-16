import unittest
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

class TestCLIContract(unittest.TestCase):
    def test_help_contract(self):
        cmd = [sys.executable, "-m", "chakrview", "chat", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("--checkpoint", out)
        self.assertIn("--memory {on,off}", out)
        self.assertIn("--debug-prompt", out)
        self.assertIn("--test-mode", out)

    def test_memory_on_mode(self):
        cmd = [sys.executable, "-m", "chakrview", "chat", "--memory", "on"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
        out, _ = proc.communicate(input="My name is Rahul.\nWhat is my name?\n/memory\n/exit\n")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("[Hybrid Memory: Name] Your name is Rahul.", out)
        self.assertIn("- Name: Rahul", out)

    def test_memory_off_mode(self):
        cmd = [sys.executable, "-m", "chakrview", "chat", "--memory", "off"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
        out, _ = proc.communicate(input="My name is Rahul.\nWhat is my name?\n/memory\n/exit\n")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("[Hybrid Memory: Name]", out)
        self.assertIn("Memory mode is currently DISABLED", out)

    def test_debug_prompt(self):
        cmd = [sys.executable, "-m", "chakrview", "chat", "--debug-prompt"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
        out, _ = proc.communicate(input="Hello assistant\n/exit\n")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("[DEBUG PROMPT]", out)
        self.assertIn("Token Count:", out)
        self.assertIn("[END DEBUG PROMPT]", out)

if __name__ == "__main__":
    unittest.main()
