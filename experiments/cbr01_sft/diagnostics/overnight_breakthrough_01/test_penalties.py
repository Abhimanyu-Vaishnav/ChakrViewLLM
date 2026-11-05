from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig
from pathlib import Path
import torch

runner = DirectNeuralRunner(Path('experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt'))
tok = runner.tokenizer

def run_chat_simulation(cfg):
    history = []
    turns = [
        'Hello ChakrView!',
        'How are you?',
        'What is your name?',
        'What can you do?',
        'My name is Abhimanyu and my project is ChakrView.',
        'What is my project called?'
    ]
    outputs = []
    for u in turns:
        prompt_parts = []
        for t in history[-4:]:
            prompt_parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
        prompt_parts.append(f"User: {u}\nAssistant:")
        prompt = '\n'.join(prompt_parts)
        
        p_toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(p_toks)
        gen = []
        with torch.no_grad():
            for step in range(48):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(next_tok)
                curr.append(next_tok)
                if next_tok == 1:
                    break
        resp = tok.decode(gen, errors='replace').strip()
        history.append({'user': u, 'assistant': resp})
        outputs.append((u, resp))
    return outputs

print('--- Greedy (no penalty) ---')
for u, r in run_chat_simulation(SamplingConfig(temperature=0.0)):
    print(f'User: {u}')
    print(f'ChakrView: {r}')
    print('-'*30)

print('\n--- Repetition Penalty 1.2 + No-repeat 3-gram ---')
for u, r in run_chat_simulation(SamplingConfig(temperature=0.0, repetition_penalty=1.2, no_repeat_ngram_size=3)):
    print(f'User: {u}')
    print(f'ChakrView: {r}')
    print('-'*30)

