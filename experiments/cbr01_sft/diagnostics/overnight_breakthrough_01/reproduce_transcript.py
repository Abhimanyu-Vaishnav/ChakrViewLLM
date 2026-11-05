from scripts.run_m3_evaluations import DirectNeuralRunner
from pathlib import Path

runner = DirectNeuralRunner(Path('experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt'))

history = []
turns = [
    'Hello ChakrView!',
    'How are you?',
    'What is your name?',
    'What can you do?',
    'My name is Abhimanyu and my project is ChakrView.',
    'What is my project called?'
]

print("REPRODUCING MULTI-TURN BEHAVIOR OF CBR-SFT-07 STEP 50:\n")
for u in turns:
    prompt_parts = []
    for t in history[-4:]:
        prompt_parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
    prompt_parts.append(f"User: {u}\nAssistant:")
    full_prompt = '\n'.join(prompt_parts)
    
    resp = runner.generate(prompt=full_prompt, max_new_tokens=48, temperature=0.0)
    assistant_resp = resp['text'].strip()
    history.append({'user': u, 'assistant': assistant_resp})
    print(f"User: {u}")
    print(f"Assistant: {assistant_resp}")
    print(f"[Tokens generated: {resp['token_count']}, stop_reason: {resp['stop_reason']}]")
    print("-" * 50)

