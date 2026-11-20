import json
from collections import Counter
from pathlib import Path

train_path = Path("data/sft/ucip14r/sft_train_12000.jsonl")
records = [json.loads(line) for line in open(train_path, encoding="utf-8")]

prompts = [r["prompt"].strip() for r in records]
responses = [r["response"].strip() for r in records]
pairs = [(r["prompt"].strip(), r["response"].strip()) for r in records]

print("=== UCIP-14R Training Records (12,000 total) ===")
print("Unique Prompts:", len(set(prompts)))
print("Unique Responses:", len(set(responses)))
print("Unique Pairs:", len(set(pairs)))

lengths = [len(r.split()) for r in responses]
print(f"Response Length (words): mean={sum(lengths)/len(lengths):.1f}, median={sorted(lengths)[len(lengths)//2]}, min={min(lengths)}, max={max(lengths)}")

openings = [" ".join(r.split()[:3]) for r in responses if len(r.split()) >= 3]
print("\nTop 10 Openings:")
for op, cnt in Counter(openings).most_common(10):
    print(f"  {cnt:4d}x ({cnt/len(responses)*100:4.1f}%): \"{op}\"")

endings = [" ".join(r.split()[-3:]) for r in responses if len(r.split()) >= 3]
print("\nTop 10 Endings:")
for ed, cnt in Counter(endings).most_common(10):
    print(f"  {cnt:4d}x ({cnt/len(responses)*100:4.1f}%): \"{ed}\"")
