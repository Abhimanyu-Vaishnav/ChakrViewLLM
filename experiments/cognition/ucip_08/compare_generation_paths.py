import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.chat import ConversationalChatCLI
from pathlib import Path
import torch

ckpt_path = Path("releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt")
runner = DirectNeuralRunner(ckpt_path)
cli = ConversationalChatCLI(ckpt_path, memory_enabled=False)

test_prompts = [
    "Hello ChakrView. How are you?",
    "What does a CPU do?",
    "नमस्ते! आप क्या कर सकते हैं?",
    "Bhai, machine learning ko simple language mein samjhao.",
    "If I have 5 apples and give 2 to my friend, how many apples remain?",
    "Say exactly: The sky is blue.",
    "What is the difference between RAM and storage?",
]

for p in test_prompts:
    print("=" * 60)
    print("USER PROMPT:", p)
    
    # 1. Direct evaluator style (pure greedy argmax, no repetition penalty, no n-gram block)
    p_eval = f"User: {p}\nAssistant:"
    p_ids = runner.tokenizer.encode(p_eval, add_bos=True, add_eos=False)
    curr = list(p_ids)
    out_eval = []
    with torch.no_grad():
        for _ in range(48):
            t = torch.tensor([curr])
            logits = runner.model(t)[0, -1, :]
            tok = torch.argmax(logits).item()
            if tok == 1:
                break
            out_eval.append(tok)
            curr.append(tok)
    txt_eval = runner.tokenizer.decode(out_eval).strip()
    
    # 2. DirectNeuralRunner generate (Sampler temp=0.0, top_k=0)
    res_runner = runner.generate(p_eval, max_new_tokens=48, temperature=0.0)
    txt_runner = res_runner["text"].strip()
    
    # 3. ConversationalChatCLI interact_step (repetition_penalty=1.15, no_repeat_ngram=3)
    cli.reset()
    txt_cli = cli.interact_step(p).strip()
    
    print("  [Evaluator 800 (Greedy)] :", repr(txt_eval))
    print("  [DirectNeuralRunner]     :", repr(txt_runner))
    print("  [Chat CLI (rep_pen=1.15)]:", repr(txt_cli))
