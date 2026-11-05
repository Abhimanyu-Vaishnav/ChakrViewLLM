import glob, json

files = glob.glob('data/sft/**/*.jsonl', recursive=True)
print(f'Found {len(files)} dataset files.')

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f]
    multiturn = [l for l in lines if 'User:' in l.get('prompt', '') and 'Assistant:' in l.get('prompt', '')]
    context_count = sum(1 for l in lines if 'context' in l.get('prompt', '').lower() or 'context' in l.get('response', '').lower())
    pct = (context_count / len(lines) * 100) if len(lines) > 0 else 0
    print(f'{fpath:45s}: {len(lines):4d} items | {len(multiturn):3d} multiturn | {context_count:3d} with context ({pct:.1f}%)')

