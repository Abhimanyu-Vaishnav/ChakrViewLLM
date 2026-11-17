import sys, torch
from pathlib import Path
REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))
from scripts.run_m3_evaluations import DirectNeuralRunner

runner = DirectNeuralRunner("releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt")
tok = runner.tokenizer

prompts = [
    "What does a CPU do?",
    "What is 2 + 2?",
    "If I have 5 apples and give 2 to my friend, how many apples remain?",
    "Say exactly: The sky is blue.",
    "Explain the difference between RAM and storage.",
    "नमस्ते! आप क्या कर सकते हैं?",
    "Bhai, machine learning ko simple language mein samjhao."
]

print("=== RAW GREEDY INFERENCE CHECK ===")
for p in prompts:
    fmt = f"User: {p}\nAssistant:"
    curr = tok.encode(fmt, add_bos=True, add_eos=False)
    g = []
    for _ in range(35):
        l = runner.model(torch.tensor([curr]))[0, -1, :]
        nxt = int(torch.argmax(l).item())
        if nxt == 1:
            break
        g.append(nxt)
        curr.append(nxt)
    text = tok.decode(g, errors="replace").strip()
    print(f"P: {p}")
    print(f"  A: {repr(text)}\n")
