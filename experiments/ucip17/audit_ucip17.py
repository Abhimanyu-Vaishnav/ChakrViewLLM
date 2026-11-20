import json
from collections import Counter
from pathlib import Path

train_path = Path("data/sft/ucip17/sft_train_12000.jsonl")
records = [json.loads(line) for line in open(train_path, encoding="utf-8")]

prompts = [r["prompt"].strip() for r in records]
responses = [r["response"].strip() for r in records]
pairs = [(r["prompt"].strip(), r["response"].strip()) for r in records]

print("=== UCIP-17 Training Records (12,000 total) ===")
print("Unique Prompts:", len(set(prompts)))
print("Unique Responses:", len(set(responses)))
print("Unique Pairs:", len(set(pairs)))

lengths = [len(r.split()) for r in responses]
print(f"Response Length (words): mean={sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)}")

# Check max frequency of any single response
top_resp, top_cnt = Counter(responses).most_common(1)[0]
print(f"Top Response count: {top_cnt}x ({top_resp[:50]}...)")
