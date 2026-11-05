from scripts.run_m3_evaluations import DirectNeuralRunner
from pathlib import Path
import torch

r7 = DirectNeuralRunner(Path('experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt'))
tok = r7.tokenizer

def clean_gen(prompt, max_toks=48):
    p_toks = tok.encode(prompt, add_bos=True, add_eos=False)
    curr = list(p_toks)
    gen = []
    with torch.no_grad():
        for step in range(max_toks):
            t_in = torch.tensor([curr], dtype=torch.long)
            logits = r7.model(t_in)[0, -1, :]
            # Anti-repetition
            if len(gen) >= 2:
                for tok_id in set(gen):
                    if logits[tok_id] > 0:
                        logits[tok_id] /= 1.25
                    else:
                        logits[tok_id] *= 1.25
            next_tok = torch.argmax(logits).item()
            gen.append(next_tok)
            curr.append(next_tok)
            if next_tok == 1:
                break
    return tok.decode(gen, errors='replace').strip()

print("SIMULATING 1-TURN SLIDING WINDOW WITH REPETITION DISCOUNT:\n")
history = []
turns = [
    'Hello ChakrView!',
    'How are you?',
    'What is your name?',
    'What can you do?',
    'Explain neural networks.',
    'Now explain it in Hinglish.'
]

for u in turns:
    prompt_parts = []
    for t in history[-1:]:
        prompt_parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
    prompt_parts.append(f"User: {u}\nAssistant:")
    p = '\n'.join(prompt_parts)
    out = clean_gen(p)
    history.append({'user': u, 'assistant': out})
    print(f"User: {u}")
    print(f"ChakrView: {out}")
    print("-" * 40)

