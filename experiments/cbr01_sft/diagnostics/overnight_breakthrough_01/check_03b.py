import json

with open("data/sft/cbr_sft_03b/splits/cbr_sft_03b_train_45.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        r = it["response"]
        print(it["id"], "startswith space:", r.startswith(" "), repr(r[:20]))

