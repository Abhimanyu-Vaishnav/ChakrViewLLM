import json
from collections import Counter, defaultdict
from pathlib import Path

path_17 = Path("data/sft/ucip17/sft_master_15000.jsonl")
records = [json.loads(line) for line in open(path_17, encoding="utf-8")]

train_recs = [r for r in records if "sft_train" in r["id"] or records.index(r) < 12000] # first 12k is train
train_recs = records[:12000]

prompts = [r["prompt"].strip() for r in train_recs]
responses = [r["response"].strip() for r in train_recs]
pairs = [(p, resp) for p, resp in zip(prompts, responses)]

print(f"Total Train Records: {len(train_recs)}")
print(f"Unique Prompts: {len(set(prompts))} ({len(set(prompts))/len(prompts)*100:.1f}%)")
print(f"Unique Responses: {len(set(responses))} ({len(set(responses))/len(responses)*100:.1f}%)")
print(f"Unique Pairs: {len(set(pairs))} ({len(set(pairs))/len(pairs)*100:.1f}%)")

exact_dupes = len(responses) - len(set(responses))
print(f"Exact Duplicate Responses: {exact_dupes} ({exact_dupes/len(responses)*100:.1f}%)")

# Category breakdown
by_cat = defaultdict(list)
by_lang = defaultdict(list)
for r in train_recs:
    by_cat[r["category"]].append(r["response"].strip())
    by_lang[r["language"]].append(r["response"].strip())

print("\n--- Unique Responses by Category ---")
for cat, resps in sorted(by_cat.items()):
    u = len(set(resps))
    print(f"  {cat:8s}: {u:4d}/{len(resps):4d} unique ({u/len(resps)*100:5.1f}%)")

print("\n--- Unique Responses by Language ---")
for lang, resps in sorted(by_lang.items()):
    u = len(set(resps))
    print(f"  {lang:8s}: {u:4d}/{len(resps):4d} unique ({u/len(resps)*100:5.1f}%)")

print("\n--- Top 10 Repeated Responses ---")
for r, cnt in Counter(responses).most_common(10):
    print(f"  {cnt:4d}x: {r[:70]}")

openings = [" ".join(r.split()[:3]) for r in responses if len(r.split()) >= 3]
print("\n--- Top 10 Openings ---")
for op, cnt in Counter(openings).most_common(10):
    print(f"  {cnt:4d}x: {op}")

lens = [len(r.split()) for r in responses]
print(f"\nResponse Length: mean={sum(lens)/len(lens):.1f}, min={min(lens)}, max={max(lens)}")
