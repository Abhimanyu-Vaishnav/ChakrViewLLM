from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig
from pathlib import Path
import torch

runner = DirectNeuralRunner(Path('experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt'))
tok = runner.tokenizer

prompts = [
    ('CBR07-0107 (Rohit)', 'User: My name is Rohit.\nAssistant: Nice to meet you, Rohit! How can I help you today?\nUser: What is my name?\nAssistant:'),
    ('CBR07-0114 (16GB RAM)', 'User: My server has 16GB of RAM.\nAssistant: Noted, 16GB RAM is a solid configuration for local CPU workloads.\nUser: How much RAM does my server have?\nAssistant:'),
    ('CBR07-0116 (github repo)', 'User: The project repository is at github.com/chakrview.\nAssistant: Noted, the repository location is recorded.\nUser: Where is the repository located?\nAssistant:')
]

cfg = SamplingConfig(temperature=0.0)

for label, p in prompts:
    p_toks = tok.encode(p, add_bos=True, add_eos=False)
    curr = list(p_toks)
    gen = []
    with torch.no_grad():
        for step in range(32):
            t_in = torch.tensor([curr], dtype=torch.long)
            logits = runner.model(t_in)[0, -1, :]
            next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
            gen.append(next_tok)
            curr.append(next_tok)
            if next_tok == 1:
                break
    out = tok.decode(gen, errors='replace').strip()
    print(f'[{label}]')
    print('  Output:', repr(out))

