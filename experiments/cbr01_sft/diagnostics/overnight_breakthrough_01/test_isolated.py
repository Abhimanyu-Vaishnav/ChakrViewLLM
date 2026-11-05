import os, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner

runner = DirectNeuralRunner(REPO_ROOT / 'experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt')

isolated_prompts = [
    'User: Hello ChakrView!\nAssistant:',
    'User: How are you?\nAssistant:',
    'User: What is your name?\nAssistant:',
    'User: What can you do?\nAssistant:',
    'User: My name is Abhimanyu and my project is ChakrView.\nAssistant:',
    'User: What is my project called?\nAssistant:'
]

print('ISOLATED SINGLE-TURN OUTPUTS (NO HISTORY):\n')
for p in isolated_prompts:
    resp = runner.generate(prompt=p, max_new_tokens=48, temperature=0.0)
    print(f'Prompt: {repr(p)}')
    print(f"Response: {repr(resp['text'].strip())}")
    print(f"Stop reason: {resp['stop_reason']}, Tokens: {resp['token_count']}")
    print('-' * 40)
